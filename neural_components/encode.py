"""
encode.py
---------
Converts a Board + RatBelief into a flat float32 numpy array.

We use the flat vector approach but "cheat" by pre-computing things the
network would otherwise have to discover on its own. This means the network
trains faster and needs fewer games to reach good play.

Output: flat vector of length N_FEATURES_1D = 706

Layout
------
Spatial layers (each is an 8x8 grid flattened to 64 floats):
  [0]    Layer 0  : My position              — one-hot
  [1]    Layer 1  : Opponent position        — one-hot
  [2]    Layer 2  : Blocked squares          — binary
  [3]    Layer 3  : Empty/space squares      — binary
  [4]    Layer 4  : Primed squares           — binary
  [5]    Layer 5  : Carpeted squares         — binary
  [6]    Layer 6  : Potential reward heatmap — float, normalised to [0,1]
  [7]    Layer 7  : Rat belief map           — float, sums to 1.0
  [8]    Layer 8  : Time ratio               — scalar broadcast to 64 floats
  [9]    Layer 9  : Turn ratio               — scalar broadcast to 64 floats
  [10]   Layer 10 : Score delta              — scalar broadcast to 64 floats

Hand-engineered scalar features (appended after the spatial layers):
  [704]  Longest carpet roll available from current position  (normalised /7)
  [705]  Number of primed squares adjacent to my position     (normalised /4)

Total: 11 * 64 + 2 = 706 floats
"""

import numpy as np
from game.enums import (
    BOARD_SIZE, Direction, CARPET_POINTS_TABLE,
    loc_after_direction, ALLOWED_TIME, MAX_TURNS_PER_PLAYER, Cell
)

N_LAYERS  = 11
BOARD_H   = BOARD_SIZE          # 8
BOARD_W   = BOARD_SIZE          # 8
N_CELLS   = BOARD_H * BOARD_W  # 64

N_SPATIAL       = N_LAYERS * N_CELLS   # 704
N_HAND_FEATURES = 2
N_FEATURES_1D   = N_SPATIAL + N_HAND_FEATURES  # 706


def encode_state(board, rat_belief) -> np.ndarray:
    """
    Parameters
    ----------
    board      : Board      — current game state (your perspective)
    rat_belief : RatBelief  — the HMM belief object from rat_belief.py

    Returns
    -------
    np.ndarray of shape (706,), dtype float32
    """
    vec = np.zeros(N_FEATURES_1D, dtype=np.float32)

    # ------------------------------------------------------------------
    # Read bitmasks once up front — much faster than 64x get_cell() calls
    # ------------------------------------------------------------------
    primed_mask  = board._primed_mask
    carpet_mask  = board._carpet_mask
    blocked_mask = board._blocked_mask
    # Space = any cell that is none of the above
    space_mask   = ~(primed_mask | carpet_mask | blocked_mask)

    # Helper: write a bitmask as 64 floats into a slice of vec
    def write_mask(layer_idx, mask):
        offset = layer_idx * N_CELLS
        for i in range(N_CELLS):
            vec[offset + i] = 1.0 if (mask & (1 << i)) else 0.0

    # ------------------------------------------------------------------
    # Layer 0: My position (one-hot)
    # ------------------------------------------------------------------
    mx, my = board.player_worker.get_location()
    my_idx = my * BOARD_SIZE + mx
    vec[0 * N_CELLS + my_idx] = 1.0

    # ------------------------------------------------------------------
    # Layer 1: Opponent position (one-hot)
    # ------------------------------------------------------------------
    ox, oy = board.opponent_worker.get_location()
    opp_idx = oy * BOARD_SIZE + ox
    vec[1 * N_CELLS + opp_idx] = 1.0

    # ------------------------------------------------------------------
    # Layers 2-5: Board cell type grids
    # ------------------------------------------------------------------
    write_mask(2, blocked_mask)
    write_mask(3, space_mask)
    write_mask(4, primed_mask)
    write_mask(5, carpet_mask)

    # ------------------------------------------------------------------
    # Layer 6: Potential reward heatmap
    #
    # For every cell, what is the best carpet score from a primed run
    # that passes through it? Tells the network which cells are "hot"
    # — worth moving toward because they unlock big carpet rolls.
    # Normalised by the max possible carpet score (21 for length 7).
    # ------------------------------------------------------------------
    heatmap = _compute_reward_heatmap(primed_mask)
    vec[6 * N_CELLS : 7 * N_CELLS] = heatmap.flatten()

    # ------------------------------------------------------------------
    # Layer 7: Rat belief map (64 floats summing to ~1.0)
    # ------------------------------------------------------------------
    if rat_belief is not None:
        vec[7 * N_CELLS : 8 * N_CELLS] = np.array(rat_belief.belief, dtype=np.float32)
    else:
        vec[7 * N_CELLS : 8 * N_CELLS] = 1.0 / N_CELLS

    # ------------------------------------------------------------------
    # Layers 8-10: Scalar globals, each broadcast across 64 cells
    #
    # We fill the entire 64-float slice with one repeated value.
    # This keeps the input shape uniform — every "column" in the flat
    # vector corresponds to one cell, including the global features.
    # ------------------------------------------------------------------

    # Layer 8: Time ratio — how much of our 4 minutes is left [0, 1]
    time_ratio = float(np.clip(board.player_worker.time_left / ALLOWED_TIME, 0.0, 1.0))
    vec[8 * N_CELLS : 9 * N_CELLS] = time_ratio

    # Layer 9: Turn ratio — turns remaining out of 40 [0, 1]
    turn_ratio = float(np.clip(board.player_worker.turns_left / MAX_TURNS_PER_PLAYER, 0.0, 1.0))
    vec[9 * N_CELLS : 10 * N_CELLS] = turn_ratio

    # Layer 10: Score delta — are we winning or losing? normalised to [-1, 1]
    score_delta = float(np.clip(
        (board.player_worker.get_points() - board.opponent_worker.get_points()) / 100.0,
        -1.0, 1.0
    ))
    vec[10 * N_CELLS : 11 * N_CELLS] = score_delta

    # ------------------------------------------------------------------
    # Hand-engineered feature 1: Longest carpet roll from current position
    #
    # We check all 4 directions and count how many consecutive primed
    # squares exist starting from the cell adjacent to us. The longest
    # of these is the carpet roll we could make right now.
    # Normalised by 7 (the max possible roll length).
    #
    # Why hand-engineer this: the network would eventually learn to read
    # layers 4+5 and figure this out itself, but it requires learning
    # to "look ahead" spatially — much harder than just being told the answer.
    # ------------------------------------------------------------------
    my_pos = (mx, my)
    longest_roll = _longest_carpet_roll(my_pos, primed_mask, board)
    vec[N_SPATIAL] = longest_roll / 7.0   # index 704

    # ------------------------------------------------------------------
    # Hand-engineered feature 2: Adjacent primed squares count
    #
    # How many of my 4 neighbours are primed? This tells the network
    # whether I'm "in position" to start a carpet roll without moving.
    # Normalised by 4 (maximum possible adjacent primed squares).
    #
    # Why hand-engineer this: adjacent relationships require the network
    # to learn spatial convolution implicitly from the flat vector.
    # Giving it directly removes that burden.
    # ------------------------------------------------------------------
    adj_primed = _adjacent_primed_count(my_pos, primed_mask)
    vec[N_SPATIAL + 1] = adj_primed / 4.0   # index 705

    return vec


# ---------------------------------------------------------------------------
# Hand-engineered feature helpers
# ---------------------------------------------------------------------------

def _longest_carpet_roll(pos, primed_mask, board) -> int:
    """
    Find the longest carpet roll available from pos in any direction
    ON THIS TURN ONLY.

    Rules for a valid carpet roll of length k:
      - k consecutive primed squares must exist starting from the cell
        immediately adjacent to pos in that direction.
      - None of those k squares may be occupied by a worker RIGHT NOW.

    Important: worker occupancy is TEMPORARY. A worker standing on a
    primed square only blocks THIS turn's roll — next turn they may have
    moved, opening the path again. We never mutate any state here, so
    this is purely a snapshot check for the current board position.

    After a carpet roll is made, those squares become carpeted (not primed),
    and either worker CAN step onto carpeted squares freely. So this
    function only matters for planning the current move.

    Returns an int in [0, 7].
    """
    # Build worker occupancy mask from the positions we were given —
    # we use pos (already extracted from the board) plus the opponent.
    # We do NOT call get_location() again to avoid reading stale state
    # if the board is a forecast copy.
    px, py  = pos
    player_bit = 1 << (py * BOARD_SIZE + px)

    ox, oy  = board.opponent_worker.get_location()
    opp_bit = 1 << (oy * BOARD_SIZE + ox)

    # Combined mask: squares blocked by either worker THIS turn only.
    # This mask is local to this function call and not stored anywhere.
    occupied_this_turn = player_bit | opp_bit

    best = 0
    for direction in Direction:
        count   = 0
        current = pos
        for _ in range(BOARD_SIZE - 1):   # max roll length = 7
            current = loc_after_direction(current, direction)
            cx, cy  = current

            # Stop if we've left the board
            if cx < 0 or cx >= BOARD_SIZE or cy < 0 or cy >= BOARD_SIZE:
                break

            bit = 1 << (cy * BOARD_SIZE + cx)

            # Square must be primed AND not occupied by any worker right now.
            # Once this roll is made these become carpet — future rolls and
            # plain steps can land on carpet freely. That's handled elsewhere.
            if not (primed_mask & bit) or (occupied_this_turn & bit):
                break

            count += 1

        best = max(best, count)
    return best


def _adjacent_primed_count(pos, primed_mask) -> int:
    """
    Count how many of the 4 cardinal neighbours of pos are primed.
    Returns an int in [0, 4].
    """
    count = 0
    for direction in Direction:
        nx, ny = loc_after_direction(pos, direction)
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            bit = 1 << (ny * BOARD_SIZE + nx)
            if primed_mask & bit:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Reward heatmap helper
# ---------------------------------------------------------------------------

def _compute_reward_heatmap(primed_mask) -> np.ndarray:
    """
    Build an 8x8 float grid where each cell's value is the carpet score
    of the best primed run that passes through it, normalised to [0, 1].

    Scans horizontal and vertical runs of consecutive primed cells,
    looks up their length in CARPET_POINTS_TABLE, and assigns the score
    to every cell in the run. Keeps the max across both directions.
    """
    heatmap = np.zeros((BOARD_H, BOARD_W), dtype=np.float32)

    def is_primed(x, y):
        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
            return False
        return bool(primed_mask & (1 << (y * BOARD_SIZE + x)))

    # Horizontal runs
    for y in range(BOARD_H):
        x = 0
        while x < BOARD_W:
            if is_primed(x, y):
                start = x
                while x < BOARD_W and is_primed(x, y):
                    x += 1
                run_len = x - start
                points  = float(CARPET_POINTS_TABLE.get(run_len, -1))
                if points > 0:
                    for rx in range(start, x):
                        heatmap[y, rx] = max(heatmap[y, rx], points)
            else:
                x += 1

    # Vertical runs
    for x in range(BOARD_W):
        y = 0
        while y < BOARD_H:
            if is_primed(x, y):
                start = y
                while y < BOARD_H and is_primed(x, y):
                    y += 1
                run_len = y - start
                points  = float(CARPET_POINTS_TABLE.get(run_len, -1))
                if points > 0:
                    for ry in range(start, y):
                        heatmap[ry, x] = max(heatmap[ry, x], points)
            else:
                y += 1

    max_score = float(max(CARPET_POINTS_TABLE.values()))  # 21
    heatmap  /= max_score
    return heatmap
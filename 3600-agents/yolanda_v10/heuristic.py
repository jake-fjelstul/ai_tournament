import numpy as np
from .move_gen import carpet_rolls

WEIGHTS = {'sd': 1.0, 'rr': 0.8, 'cp': 0.8, 'pq': 0.3}
CARPET_SCORE = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}
_rp_cache = {}

DIST_MATRIX = np.zeros((64, 64), dtype=np.float32)
for i in range(64):
    ix, iy = i % 8, i // 8
    for j in range(64):
        jx, jy = j % 8, j // 8
        DIST_MATRIX[i, j] = abs(ix - jx) + abs(iy - jy)

def compute_cell_potential(blocked_mask):
    """Counts contiguous unblocked cells in cardinal directions. Runs ONCE at init."""
    potential = np.zeros(64, dtype=np.float32)
    for i in range(64):
        if blocked_mask & (1 << i): continue
        x, y = i % 8, i // 8
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                j = ny*8 + nx
                if blocked_mask & (1 << j): break
                potential[i] += 1
                nx += dx; ny += dy
    return potential

def reachable_potential(worker_pos, cell_potential, turns_left):
    w_idx = worker_pos[1] * 8 + worker_pos[0]
    
    # Grab the precomputed distances for the worker's current cell
    dists = DIST_MATRIX[w_idx]
    
    # Use NumPy masking to only evaluate cells within range
    valid_mask = dists <= turns_left
    decay = 1.0 / (1.0 + dists[valid_mask])
    
    # Fast vectorized multiplication and sum
    return np.sum(cell_potential[valid_mask] * decay)

def evaluate(board, belief, cell_potential, is_max=True, w=None):
    if w is None:
        w = {'sd': 3.0, 'rr': 0.2, 'cp': 0.8, 'pq': 0.3}

    if not is_max:
        board.reverse_perspective()

    turns_left = board.player_worker.turns_left

    if turns_left <= 20:
        # w['rev'] = min(w['rev'] * (1.0 + (8 - turns_left) * 0.15), 1.2)
        w['cp']  = w['cp'] * max(0.3, turns_left / 40)

    sd = board.player_worker.get_points() - board.opponent_worker.get_points()

    # p = belief.belief.max()
    # rev = 4.0 * p - 2.0 * (1.0 - p)

    rolls = carpet_rolls(board)
    raw_rr = max((CARPET_SCORE.get(r.roll_length, 0) for r in rolls), default=0)
    rr = max(0, raw_rr)

    cp = reachable_potential(board.player_worker.position, cell_potential, turns_left)
    i = board.player_worker.position[1]*8 + board.player_worker.position[0]
    pq = cell_potential[i]

    board.reverse_perspective()
    opp_rolls = carpet_rolls(board) 
    board.reverse_perspective() # Flip it back immediately so we don't break the state
    
    opp_rr = max((CARPET_SCORE.get(r.roll_length, 0) for r in opp_rolls), default=0)
    opp_rr = max(0, opp_rr)

    runway_bonus = 0.0
    wx, wy = board.player_worker.position
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        cx, cy = wx + dx, wy + dy
        if 0 <= cx < 8 and 0 <= cy < 8:
            bit = 1 << (cy * 8 + cx)
            if board._primed_mask & bit:
                runway_bonus += 0.2 # Dropped from 1.5

    if turns_left <= 4:
        w['cp'] = 0.0          # Future potential is worthless now
        w['pq'] = 0.0          # Current cell potential is worthless
        runway_bonus = 0.0     # Stop building runways
        w['sd'] = 2.5          # Hyper-scale immediate score differences
        w['rr'] = 1.5

    # Add the runway_bonus to your final calculation
    val = w['sd']*sd + w['rr']*rr + w['cp']*cp + w['pq']*pq - (0.8 * opp_rr) + runway_bonus

    if not is_max:
        board.reverse_perspective()

    return val

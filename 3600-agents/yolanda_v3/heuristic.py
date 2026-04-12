import numpy as np
from .move_gen import carpet_rolls

# WEIGHTS = {'sd': 1.0, 'rev': 0.1, 'rr': 1.5, 'cp': 0.3, 'pq': 0.1}
WEIGHTS = {'sd': 1.0, 'rr': 1.2, 'cp': 0.8, 'pq': 0.3}
CARPET_SCORE = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}

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
    wx, wy = worker_pos
    total  = 0.0
    for i in range(64):
        rx, ry = i % 8, i // 8
        dist   = abs(rx - wx) + abs(ry - wy)
        if dist <= turns_left:
            decay  = 1.0 / (1.0 + dist)
            total += cell_potential[i] * decay
    return total

def evaluate(board, belief, cell_potential, is_max=True, w=None):
    if w is None:
        w = WEIGHTS.copy()

    turns_left = board.player_worker.turns_left

    if turns_left <= 8:
        # w['rev'] = min(w['rev'] * (1.0 + (8 - turns_left) * 0.15), 1.2)
        w['cp']  = w['cp'] * 0.5

    sd = board.player_worker.get_points() - board.opponent_worker.get_points()

    # p = belief.belief.max()
    # rev = 4.0 * p - 2.0 * (1.0 - p)

    rolls = carpet_rolls(board)
    rr = max((CARPET_SCORE.get(r.roll_length, 0) for r in rolls), default=0)

    cp = reachable_potential(board.player_worker.position, cell_potential, turns_left)
    i = board.player_worker.position[1]*8 + board.player_worker.position[0]
    pq = cell_potential[i]

    val = w['sd']*sd + w['rr']*rr + w['cp']*cp + w['pq']*pq
    return val if is_max else -val

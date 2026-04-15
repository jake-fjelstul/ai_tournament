import numpy as np
from .move_gen import carpet_rolls
import os

# Put this at the top of heuristic.py
WEIGHTS = {
    'sd': float(os.environ.get('YOLANDA_W_SD', 1.0)),
    'rr': float(os.environ.get('YOLANDA_W_RR', 0.8)),
    'cp': float(os.environ.get('YOLANDA_W_CP', 0.8)),
    'pq': float(os.environ.get('YOLANDA_W_PQ', 0.3))
}
#WEIGHTS = {'sd': 1.0, 'rr': 0.8, 'cp': 0.8, 'pq': 0.3}
CARPET_SCORE = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}

DIST_MATRIX = np.zeros((64, 64), dtype=np.float32)
for i in range(64):
    ix, iy = i % 8, i // 8
    for j in range(64):
        jx, jy = j % 8, j // 8
        DIST_MATRIX[i, j] = abs(ix - jx) + abs(iy - jy)

# Cache for heuristic evaluations
_eval_cache = {}

def compute_cell_potential(blocked_mask):
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
    """Reverted the hard opponent mask to prevent evasion behavior."""
    w_idx = worker_pos[1] * 8 + worker_pos[0]
    dists = DIST_MATRIX[w_idx]
    
    valid_mask = dists <= turns_left
    decay = 1.0 / (1.0 + dists[valid_mask])
    
    return np.sum(cell_potential[valid_mask] * decay)

def evaluate(board, belief, cell_potential, is_max=True, w=None):
    if w is None:
        w = {'sd': 3.0, 'rr': 0.2, 'cp': 0.8, 'pq': 0.3}

    turns_left = board.player_worker.turns_left
    sd = board.player_worker.get_points() - board.opponent_worker.get_points()

    # FIX: Added turns_left to the hash. Horizon scaling requires turns_left to be unique!
    state_hash = hash((board.player_worker.position, board.opponent_worker.position, board._primed_mask, board._carpet_mask, is_max, sd, turns_left))
    if state_hash in _eval_cache:
        return _eval_cache[state_hash]

    if not is_max:
        board.reverse_perspective()

    if turns_left <= 20:
        w['cp']  = w['cp'] * max(0.3, turns_left / 40)

    rolls = carpet_rolls(board)
    raw_rr = max((CARPET_SCORE.get(r.roll_length, -5) for r in rolls), default=0)
    rr = max(0, raw_rr)

    cp = reachable_potential(board.player_worker.position, cell_potential, turns_left)
    i = board.player_worker.position[1]*8 + board.player_worker.position[0]
    pq = cell_potential[i]

    board.reverse_perspective()
    opp_rolls = carpet_rolls(board) 
    board.reverse_perspective() 
    
    opp_rr = max((CARPET_SCORE.get(r.roll_length, -5) for r in opp_rolls), default=0)
    opp_rr = max(0, opp_rr)

    runway_bonus = 0.0
    wx, wy = board.player_worker.position
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        cx, cy = wx + dx, wy + dy
        nx, ny = wx + (dx*2), wy + (dy*2)
        if 0 <= cx < 8 and 0 <= cy < 8:
            if board._primed_mask & (1 << (cy * 8 + cx)):
                if 0 <= nx < 8 and 0 <= ny < 8:
                    n_bit = 1 << (ny * 8 + nx)
                    # Ensure we are actually building towards empty space
                    if not (board._blocked_mask & n_bit) and not (board._carpet_mask & n_bit):
                        runway_bonus += 0.2

    if turns_left <= 4:
        w['cp'] = 0.0          
        w['pq'] = 0.0          
        runway_bonus = 0.0     
        w['sd'] = 2.5          
        w['rr'] = 1.5

    val = w['sd']*sd + w['rr']*rr + w['cp']*cp + w['pq']*pq - (0.8 * opp_rr) + runway_bonus

    if not is_max:
        board.reverse_perspective()

    _eval_cache[state_hash] = val
    return val

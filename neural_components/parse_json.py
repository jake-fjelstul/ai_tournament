"""
parse_json.py
-------------
Converts 121k bytefight JSON files into a sharded training dataset.

4.8M turns × 706 floats × 4 bytes ≈ 13.5 GB — too big for RAM.
We save as shards of 50k turns each (~140 MB per shard, ~96 shards total).
Training reads one shard at a time.

Usage:
    python3 parse_json.py --json_dir bytefight_cs3600_sp2026/ --out_dir dataset_shards/

Output: dataset_shards/shard_0000.npz, shard_0001.npz, ... + meta.npy
"""

import os, sys, json, argparse, glob
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'engine'))
sys.path.insert(0, os.path.join(ROOT, '3600-agents'))

from game.enums import BOARD_SIZE, Direction, Cell
from game.move import Move
from game.board import Board
from yolanda_31.encode import encode_state
from yolanda_31.rat_belief import RatBelief
from yolanda_31.collect import move_to_index

MAX_MARGIN = 100.0
DIR_MAP = {
    (0,-1): Direction.UP, (0,1): Direction.DOWN,
    (-1,0): Direction.LEFT, (1,0): Direction.RIGHT,
}
T_UNIFORM = [[1/64]*64 for _ in range(64)]


def reconstruct_move(lb, pos_before, pos_after, new_carpets):
    if lb == 'search':
        return Move.search((0, 0))
    dx = pos_after[0] - pos_before[0]
    dy = pos_after[1] - pos_before[1]
    if lb == 'carpet' and new_carpets:
        cdx = new_carpets[0][0] - pos_before[0]
        cdy = new_carpets[0][1] - pos_before[1]
        d   = DIR_MAP.get((cdx, cdy), Direction.RIGHT)
        return Move.carpet(d, len(new_carpets))
    d = DIR_MAP.get((dx, dy), Direction.RIGHT)
    return Move.prime(d) if lb == 'prime' else Move.plain(d)


def build_board(match, turn, is_a):
    board = Board()
    for bx, by in match['blocked_positions']:
        board.set_cell((bx, by), Cell.BLOCKED)

    primed, carpeted = set(), set()
    for t in range(turn):
        lb = match['left_behind'][t]
        nc = match['new_carpets'][t]
        if lb == 'prime':
            pos = match['a_pos'][t] if t % 2 == 0 else match['b_pos'][t]
            primed.add(tuple(pos))
        elif lb == 'carpet':
            for cx, cy in nc:
                carpeted.add((cx, cy))
                primed.discard((cx, cy))

    for px, py in primed:   board.set_cell((px, py), Cell.PRIMED)
    for cx, cy in carpeted: board.set_cell((cx, cy), Cell.CARPET)

    ap, bp = tuple(match['a_pos'][turn]), tuple(match['b_pos'][turn])
    if is_a:
        board.player_worker.position     = ap
        board.player_worker.points       = match['a_points'][turn]
        board.player_worker.turns_left   = match['a_turns_left'][turn]
        board.player_worker.time_left    = match['a_time_left'][turn]
        board.opponent_worker.position   = bp
        board.opponent_worker.points     = match['b_points'][turn]
        board.opponent_worker.turns_left = match['b_turns_left'][turn]
        board.opponent_worker.time_left  = match['b_time_left'][turn]
    else:
        board.player_worker.position     = bp
        board.player_worker.points       = match['b_points'][turn]
        board.player_worker.turns_left   = match['b_turns_left'][turn]
        board.player_worker.time_left    = match['b_time_left'][turn]
        board.opponent_worker.position   = ap
        board.opponent_worker.points     = match['a_points'][turn]
        board.opponent_worker.turns_left = match['a_turns_left'][turn]
        board.opponent_worker.time_left  = match['a_time_left'][turn]
    return board


def parse_match(match):
    states, moves, outcomes = [], [], []
    n   = min(match['turn_count'], len(match['left_behind']))
    fa, fb = match['a_points'][-1], match['b_points'][-1]
    ma  = float(np.clip((fa - fb) / MAX_MARGIN, -1.0, 1.0))
    mb  = float(np.clip((fb - fa) / MAX_MARGIN, -1.0, 1.0))

    belief = RatBelief(T_UNIFORM)
    belief.initialize()

    for t in range(n - 1):
        lb    = match['left_behind'][t]
        is_a  = (t % 2 == 0)
        if lb == 'search':
            continue
        try:
            board = build_board(match, t, is_a)
            pb    = match['a_pos'][t]   if is_a else match['b_pos'][t]
            pa    = match['a_pos'][t+1] if is_a else match['b_pos'][t+1]
            m     = reconstruct_move(lb, pb, pa, match['new_carpets'][t])
            idx   = move_to_index(m)
            if idx < 0: continue
            states.append(encode_state(board, belief))
            moves.append(idx)
            outcomes.append(ma if is_a else mb)
        except Exception:
            continue
    return states, moves, outcomes


def parse_all(json_dir, out_dir, shard_size=50000):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    print(f"Found {len(files)} files. Parsing into shards of {shard_size} turns...")

    buf_s, buf_m, buf_o = [], [], []
    shard_idx = total_turns = skipped = 0

    def flush():
        nonlocal shard_idx
        if not buf_s: return
        path = os.path.join(out_dir, f'shard_{shard_idx:04d}.npz')
        np.savez_compressed(path,
            states   = np.array(buf_s, dtype=np.float32),
            moves    = np.array(buf_m, dtype=np.int32),
            outcomes = np.array(buf_o, dtype=np.float32))
        print(f"  Shard {shard_idx:04d}: {len(buf_s)} turns → {path}")
        shard_idx += 1
        buf_s.clear(); buf_m.clear(); buf_o.clear()

    for i, fpath in enumerate(files):
        if i % 5000 == 0:
            print(f"[{i}/{len(files)}] turns={total_turns} shards={shard_idx}")
        try:
            with open(fpath) as f:
                match = json.load(f)
            s, m, o = parse_match(match)
            buf_s.extend(s); buf_m.extend(m); buf_o.extend(o)
            total_turns += len(s)
            if len(buf_s) >= shard_size:
                flush()
        except Exception:
            skipped += 1

    flush()
    np.save(os.path.join(out_dir, 'meta.npy'),
            {'total_turns': total_turns, 'num_shards': shard_idx, 'shard_size': shard_size})
    print(f"\nDone. {total_turns} turns, {shard_idx} shards, {skipped} skipped.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--json_dir',   default='bytefight_cs3600_sp2026')
    p.add_argument('--out_dir',    default='dataset_shards')
    p.add_argument('--shard_size', type=int, default=50000)
    args = p.parse_args()
    parse_all(args.json_dir, args.out_dir, args.shard_size)

"""
collect.py
----------
Runs N games between your DataCollectionAgent and a strong opponent,
saving (state, move, outcome) data to disk after every game.
 
Usage (run from the repo root):
    python3 -m YourBotName.collect --opponent PartnerBot --games 200 --out dataset.npy
 
The saved dataset is a .npy file containing a dict:
    states   : (total_turns, 706)  float32  — encoded board states
    moves    : (total_turns,)      int32    — move index chosen
    outcomes : (total_turns,)      float32  — +1.0 win, -1.0 loss, 0.0 tie
"""
 
import argparse
import os
import pathlib
import sys
import numpy as np
from collections.abc import Callable
from typing import List, Tuple
 
# ---------------------------------------------------------------------------
# Path setup — lets us import from the engine regardless of where we're run
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()  # repo root
sys.path.insert(0, str(ROOT))
 
from gameplay import play_game
from game.enums import (MoveType, Direction, BOARD_SIZE,
                        MAX_TURNS_PER_PLAYER, CARPET_POINTS_TABLE,
                        Cell, loc_after_direction)
from game.move import Move
from game.enums import Result
 
from .encode import encode_state, N_FEATURES_1D
from .rat_belief import RatBelief
 
# ---------------------------------------------------------------------------
# Move vocabulary — every possible move as a fixed integer index
# ---------------------------------------------------------------------------
# The network outputs a probability over N_MOVES possible actions.
# We need a consistent mapping from Move objects to integers.
#
# Layout:
#   [0:4]    plain  — 4 directions
#   [4:8]    prime  — 4 directions
#   [8:36]   carpet — 4 directions × 7 roll lengths
#   [36:100] search — 64 cells (one per board position)
#   Total: 100 moves
 
def _build_move_vocab() -> List[Move]:
    vocab = []
    for d in Direction:
        vocab.append(Move.plain(d))
    for d in Direction:
        vocab.append(Move.prime(d))
    for d in Direction:
        for roll in range(1, BOARD_SIZE):
            vocab.append(Move.carpet(d, roll))
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            vocab.append(Move.search((x, y)))
    return vocab
 
MOVE_VOCAB = _build_move_vocab()
N_MOVES    = len(MOVE_VOCAB)  # 100
 
 
def move_to_index(m: Move) -> int:
    """Convert a Move to its integer index in MOVE_VOCAB. Returns -1 if not found."""
    for i, vm in enumerate(MOVE_VOCAB):
        if vm.move_type != m.move_type:
            continue
        if vm.direction != m.direction:
            continue
        if m.move_type == MoveType.CARPET and vm.roll_length != m.roll_length:
            continue
        if m.move_type == MoveType.SEARCH and vm.search_loc != m.search_loc:
            continue
        return i
    return -1
 
 
def index_to_move(idx: int) -> Move:
    return MOVE_VOCAB[idx]
 
 
# ---------------------------------------------------------------------------
# DataCollectionAgent
# ---------------------------------------------------------------------------
 
class DataCollectionAgent:
    """
    Plays games using the greedy strategy and records every
    (state_vector, move_index) pair. After each game, call
    save_game(outcome) to label and persist the data.
 
    The engine calls __init__ once before the game, then play() each turn.
    save_game() is called by OUR collection script (not the engine) after
    play_game() returns.
    """
 
    def __init__(self, board, transition_matrix=None, time_left: Callable = None,
                 out_path: str = "dataset.npy"):
        self.out_path = out_path
 
        if transition_matrix is not None:
            self.rat_belief = RatBelief(transition_matrix)
            self.rat_belief.initialize()   # runs the 1000-step headstart simulation
        else:
            self.rat_belief = None
 
        # Buffer for current game — cleared after each save_game() call
        self._game_states : List[np.ndarray] = []
        self._game_moves  : List[int]        = []
 
    def commentate(self) -> str:
        return f"Recorded {len(self._game_states)} turns."
 
    def play(self, board, sensor_data: Tuple, time_left: Callable) -> Move:
        noise, estimated_dist = sensor_data
        worker_pos = board.player_worker.get_location()
 
        # --- update rat belief (using partner's RatBelief method names) ---
        if self.rat_belief is not None:
            self.rat_belief.predict()
            self.rat_belief.update_noise(noise, board)
            self.rat_belief.update_distance(estimated_dist, worker_pos)
 
            if hasattr(board, 'opponent_search') and board.opponent_search[0] is not None:
                self.rat_belief.update_opponent_search(board.opponent_search)
            if hasattr(board, 'player_search') and board.player_search[0] is not None:
                self.rat_belief.update_opponent_search(board.player_search)
 
        # --- encode state BEFORE choosing move ---
        state_vec = encode_state(board, self.rat_belief)
 
        # Use partner's greedy_move directly — no need to duplicate the logic
        from .agent import greedy_move
        chosen_move = greedy_move(board, self.rat_belief)
 
        # --- record ---
        idx = move_to_index(chosen_move)
        if idx >= 0:
            self._game_states.append(state_vec)
            self._game_moves.append(idx)
 
        return chosen_move
 
    def save_game(self, outcome: float):
        """
        Call this after play_game() returns.
 
        Parameters
        ----------
        outcome : +1.0 = win, -1.0 = loss, 0.0 = tie
        """
        if not self._game_states:
            self._game_states = []
            self._game_moves  = []
            return
 
        n        = len(self._game_states)
        states   = np.stack(self._game_states)                  # (n, 706)
        moves    = np.array(self._game_moves,  dtype=np.int32)  # (n,)
        outcomes = np.full(n, outcome,         dtype=np.float32)# (n,)
 
        # Append to existing dataset or create new one
        if os.path.exists(self.out_path):
            existing = np.load(self.out_path, allow_pickle=True).item()
            states   = np.concatenate([existing['states'],   states])
            moves    = np.concatenate([existing['moves'],    moves])
            outcomes = np.concatenate([existing['outcomes'], outcomes])
 
        np.save(self.out_path, {'states': states, 'moves': moves, 'outcomes': outcomes})
 
        print(f"  Saved {n} turns (outcome={outcome:+.0f}). "
              f"Dataset now has {len(states)} turns total.")
 
        # Reset for next game
        self._game_states = []
        self._game_moves  = []
 
 
 
 
# ---------------------------------------------------------------------------
# Collection script — this is what you actually run
# ---------------------------------------------------------------------------
 
def run_collection(play_directory: str, opponent_name: str,
                   n_games: int, out_path: str):
    """
    Run n_games and save all data to out_path.
 
    How it works:
      1. Call play_game() from the engine directly (same as run_local_agents.py)
      2. Read the winner from final_board.get_winner()
      3. Call agent_a.save_game(outcome) to persist the data
 
    We run DataCollectionAgent as player A every game.
    The opponent (your partner's strong bot) is always player B.
    """
    wins = losses = ties = 0
 
    for game_num in range(1, n_games + 1):
        print(f"Game {game_num}/{n_games} ...", end=" ", flush=True)
 
        # play_game returns final_board which has .get_winner()
        # We pass out_path through via a wrapper so DataCollectionAgent
        # can be constructed with it (the engine calls __init__ itself)
        final_board, _, _, _, _, _ = play_game(
            play_directory,   # directory for player A (us)
            play_directory,   # directory for player B (opponent)
            "YourBotName",    # <-- change to your actual folder name
            opponent_name,
            display_game=False,
            delay=0.0,
            clear_screen=False,
            record=False,
            limit_resources=False,
        )
 
        # Determine outcome from final_board
        # Result.PLAYER = we (player A) won
        # Result.ENEMY  = opponent (player B) won
        # Result.TIE    = tie
        winner = final_board.get_winner()
        if winner == Result.PLAYER:
            outcome = +1.0
            wins   += 1
            label   = "WIN"
        elif winner == Result.ENEMY:
            outcome = -1.0
            losses += 1
            label   = "LOSS"
        else:
            outcome = 0.0
            ties   += 1
            label   = "TIE"
 
        print(label)
 
        # ----------------------------------------------------------------
        # The problem: play_game() constructs the agent internally so we
        # don't have a direct reference to call save_game() on.
        #
        # Solution: DataCollectionAgent writes to a shared temp buffer
        # file each turn, and we read + label it here.
        #
        # Simpler solution for now: run a small wrapper (see below).
        # ----------------------------------------------------------------
 
    print(f"\nDone. {wins}W / {losses}L / {ties}T over {n_games} games.")
 
 
# ---------------------------------------------------------------------------
# The save_game problem — and the fix
# ---------------------------------------------------------------------------
#
# play_game() constructs your agent internally, so after it returns you
# don't have a reference to the DataCollectionAgent instance to call
# save_game() on.
#
# The fix: make DataCollectionAgent write its buffer to a TEMP file at
# the end of each game (in commentate()), then the collection script
# reads that temp file, labels it with the outcome, and appends to the
# real dataset.
#
# See DataCollectionAgentWithTempFile below — use THIS as your agent.py
# for data collection runs.
 
TEMP_BUFFER_PATH = "temp_game_buffer.npy"
 
 
class DataCollectionAgentWithTempFile(DataCollectionAgent):
    """
    Same as DataCollectionAgent but writes the unlabelled buffer to a
    temp file in commentate() so the collection script can label it.
 
    Use this as your agent.py when running collection games.
    Replace agent.py in your bot folder temporarily, run collection,
    then swap back to your real agent.py for the tournament.
    """
 
    def commentate(self) -> str:
        if not self._game_states:
            return "No turns recorded."
 
        # Write unlabelled buffer to temp file
        np.save(TEMP_BUFFER_PATH, {
            'states': np.stack(self._game_states),
            'moves':  np.array(self._game_moves, dtype=np.int32),
        })
        return f"Wrote {len(self._game_states)} turns to {TEMP_BUFFER_PATH}"
 
 
def run_collection_with_temp(play_directory: str, your_bot_name: str,
                              opponent_name: str, n_games: int, out_path: str):
    """
    The working collection loop using the temp file approach.
 
    Flow each game:
      1. play_game() runs — DataCollectionAgentWithTempFile writes temp file
                            in its commentate() call at game end
      2. We read the temp file
      3. We label every turn with the outcome
      4. We append to the real dataset
    """
    wins = losses = ties = 0
 
    # Collect all games in memory, save once at the end.
    # Loading and rewriting dataset.npy every game gets very slow
    # once you have thousands of turns — O(n) work per game = O(n^2) total.
    all_states         = []
    all_moves          = []
    all_outcomes       = []
    turns_this_session = 0
 
    for game_num in range(1, n_games + 1):
        print(f"Game {game_num}/{n_games} ...", end=" ", flush=True)
 
        # Remove stale temp file before each game
        if os.path.exists(TEMP_BUFFER_PATH):
            os.remove(TEMP_BUFFER_PATH)
 
        final_board, _, _, _, _, _ = play_game(
            play_directory,
            play_directory,
            your_bot_name,
            opponent_name,
            display_game=False,
            delay=0.0,
            clear_screen=False,
            record=False,
            limit_resources=False,
        )
 
        # Use score margin instead of +1/-1 so the network learns that
        # winning by 50 is much better than winning by 1.
        # final_board is always from player A's perspective after play_game().
        my_points  = final_board.player_worker.get_points()
        opp_points = final_board.opponent_worker.get_points()
        margin     = my_points - opp_points   # positive = we won, negative = we lost
        outcome    = float(margin)
 
        winner = final_board.get_winner()
        if winner == Result.PLAYER:
            wins  += 1
            label  = f"WIN   (+{margin})"
        elif winner == Result.ENEMY:
            losses += 1
            label   = f"LOSS  ({margin})"
        else:
            ties  += 1
            label  = f"TIE   ({margin})"
 
        print(label)
 
        # Read temp buffer written by commentate()
        if not os.path.exists(TEMP_BUFFER_PATH):
            print("  WARNING: no temp buffer found, skipping game.")
            continue
 
        buffer = np.load(TEMP_BUFFER_PATH, allow_pickle=True).item()
        n      = len(buffer['moves'])
 
        # Accumulate in memory — we save once at the very end.
        # This avoids loading and rewriting the entire dataset file
        # on every game, which gets very slow as the dataset grows.
        all_states.append(buffer['states'])                          # (n, 706)
        all_moves.append(buffer['moves'])                            # (n,)
        all_outcomes.append(np.full(n, outcome, dtype=np.float32))  # (n,)
 
        turns_this_session += n
        print(f"  +{n} turns this game. Session total: {turns_this_session}")
 
    # ------------------------------------------------------------------
    # Save everything once at the end
    # ------------------------------------------------------------------
    print(f"\nDone. {wins}W / {losses}L / {ties}T over {n_games} games.")
 
    if not all_states:
        print("No data collected.")
        return
 
    new_states   = np.concatenate(all_states)
    new_moves    = np.concatenate(all_moves)
    new_outcomes = np.concatenate(all_outcomes)
 
    # If a dataset already exists on disk, append to it
    if os.path.exists(out_path):
        existing     = np.load(out_path, allow_pickle=True).item()
        new_states   = np.concatenate([existing['states'],   new_states])
        new_moves    = np.concatenate([existing['moves'],    new_moves])
        new_outcomes = np.concatenate([existing['outcomes'], new_outcomes])
 
    np.save(out_path, {'states': new_states, 'moves': new_moves, 'outcomes': new_outcomes})
    print(f"Saved {len(new_moves)} total turns to {out_path}")
 
    if os.path.exists(TEMP_BUFFER_PATH):
        os.remove(TEMP_BUFFER_PATH)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent",  default="PartnerBot",
                        help="Folder name of the opponent bot in 3600-agents/")
    parser.add_argument("--yourbot",   default="YourBotName",
                        help="Folder name of your collection bot")
    parser.add_argument("--games",     type=int, default=100)
    parser.add_argument("--out",       default="dataset.npy")
    args = parser.parse_args()
 
    play_dir = str(ROOT / "3600-agents")
 
    run_collection_with_temp(
        play_directory=play_dir,
        your_bot_name=args.yourbot,
        opponent_name=args.opponent,
        n_games=args.games,
        out_path=args.out,
    )
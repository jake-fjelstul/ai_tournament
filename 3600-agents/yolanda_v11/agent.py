from collections.abc import Callable
from typing import List, Set, Tuple
import random

try:
    from game.board import Board
    from game.move import Move
except ImportError:
    from engine.game.board import Board
    from engine.game.move import Move

from .move_gen import all_legal_moves, carpet_rolls, prime_steps, plain_steps
from .rat_belief import RatBelief
from .search import SearchEngine
from .heuristic import compute_cell_potential

class PlayerAgent:
    def __init__(self, board, transition_matrix, time_left_func):
        self.T = transition_matrix
        self.time_left_func = time_left_func
        
        # Yolanda's personal belief
        self.belief = RatBelief(self.T)
        self.belief.initialize()
        
        # Simulated Opponent Belief
        self.opp_belief = RatBelief(self.T)
        self.opp_belief.initialize()
        
        self.search = SearchEngine()
        self.cell_potential = compute_cell_potential(board._blocked_mask)

    def play(self, board: 'Board', sensor_data: Tuple, time_left: Callable):
        try:
            self.time_left_func = time_left
            
            if sensor_data is not None:
                noise, dist = sensor_data
                
                # 1. Apply Yolanda's search from the end of her LAST turn
                if hasattr(board, 'player_search') and board.player_search[0] is not None:
                    self.belief.update_opponent_search(board.player_search)
                    self.opp_belief.update_opponent_search(board.player_search) # Opponent knows we missed
                
                # 2. Rat takes a step for opponent's turn
                self.belief.predict()
                self.opp_belief.predict()

                # 3. Apply opponent's search from their turn
                if hasattr(board, 'opponent_search') and board.opponent_search[0] is not None:
                    self.belief.update_opponent_search(board.opponent_search)
                    self.opp_belief.update_opponent_search(board.opponent_search)
                else:
                    # If they DIDN'T search, their EV was likely < 0.5. 
                    # We can decay their peak probability slightly to model their uncertainty.
                    pass 
                
                # 4. Rat takes a step for Yolanda's current turn
                self.belief.predict()
                self.opp_belief.predict()
                
                # 5. Apply fresh sensor data (ONLY FOR YOLANDA)
                self.belief.update_noise(noise, board)
                self.belief.update_distance(dist, board.player_worker.position)
                
            turns_left = board.player_worker.turns_left
            current_time_left = self.time_left_func()

            if current_time_left < 0.5:
                return greedy_move(board, self.belief)
            
            time_slice = current_time_left / max(1, turns_left)
            budget = min(time_slice * 0.95, 8.0)
            
            # Pass our belief into search
            return self.search.run(board, self.belief, budget, self.cell_potential)
            
        except Exception as e:
            print(f"CRITICAL ERROR in play(): {e}")
            from .agent import greedy_move
            return greedy_move(board, self.belief)
    
    def commentate(self):
        return "Good game!"

def greedy_move(board: Board, belief: RatBelief = None) -> Move:
    moves = all_legal_moves(board, belief)
    if not moves:
        return Move.search((0, 0))

    carps = carpet_rolls(board)
    if carps:
        best_carpet = max(carps, key=lambda m: m.roll_length)
        return best_carpet
        
    if belief is not None and belief.search_ev() > 0.5:
        bx, by = belief.best_search_target()
        return Move.search((bx, by))
        
    primes = prime_steps(board)
    if primes:
        return random.choice(primes)
        
    plains = plain_steps(board)
    if plains:
        return random.choice(plains)
        
    return moves[0]
import time
import numpy as np
from .heuristic import evaluate, CARPET_SCORE, WEIGHTS
from .move_gen import all_legal_moves
try:
    from game.enums import Result
except ImportError:
    from engine.game.enums import Result

def order_moves(moves, belief, is_max=True):
    def quick_score(move):
        if hasattr(move, 'roll_length') and move.roll_length:
            if move.roll_length == 1:
                return -100.0 
            return CARPET_SCORE.get(move.roll_length, 0) * 3
            
        if move.move_type.name == 'SEARCH': 
            return belief.search_ev() * 2 
            
        if move.move_type.name == 'PRIME': 
            return 1.0
            
        return 0.0 
        
    return sorted(moves, key=quick_score, reverse=True)

def minimax(board, belief, cell_potential, depth, alpha, beta, is_max, deadline, counter):
    counter[0] += 1
    if counter[0] % 50 == 0:  
        if time.time() >= deadline:
            raise TimeoutError()

    if board.is_game_over():
        winner = board.get_winner()
        if winner == Result.TIE: return 0, None
        elif winner == Result.PLAYER: return (float('inf') if is_max else float('-inf')), None
        elif winner == Result.ENEMY: return (float('-inf') if is_max else float('inf')), None
        else: return evaluate(board, belief, cell_potential, is_max), None

    if depth == 0:
        return evaluate(board, belief, cell_potential, is_max), None

    moves = all_legal_moves(board, belief)
    if not moves:
        return evaluate(board, belief, cell_potential, is_max), None

    ordered_moves = order_moves(moves, belief, is_max)
    best_move = ordered_moves[0]
    
    sd_multiplier = 1.0 + (board.player_worker.turns_left / 40.0)
    SD_WEIGHT = WEIGHTS.get('sd', 3.0) * sd_multiplier

    if is_max:
        max_eval = float('-inf')
        for move in ordered_moves:
            child = board.forecast_move(move)
            if child is None: continue
            child.reverse_perspective()

            if move.move_type.name == 'SEARCH':
                bx, by = belief.best_search_target()
                target_idx = by * 8 + bx
                p_find = belief.belief[target_idx]

                # HYBRID EXPECTIMINIMAX: Only branch if probability is very high
                if p_find > 0.25:
                    belief_find = belief.copy()
                    belief_find.initialize()
                    belief_find.predict()
                    val_find, _ = minimax(child, belief_find, cell_potential, depth - 1, float('-inf'), float('inf'), not is_max, deadline, counter)
                    val_find += (4.0 * SD_WEIGHT) 

                    belief_miss = belief.copy()
                    belief_miss.belief[target_idx] = 0.0
                    s = belief_miss.belief.sum()
                    belief_miss.belief = belief_miss.belief / s if s > 1e-12 else np.ones(64) / 64.0
                    belief_miss.predict()
                    val_miss, _ = minimax(child, belief_miss, cell_potential, depth - 1, float('-inf'), float('inf'), not is_max, deadline, counter)
                    val_miss -= (2.0 * SD_WEIGHT) 

                    eval_val = (p_find * val_find) + ((1.0 - p_find) * val_miss)
                else:
                    # Save depth: Treat as a miss, but add the mathematical EV
                    belief_miss = belief.copy()
                    belief_miss.belief[target_idx] = 0.0
                    s = belief_miss.belief.sum()
                    belief_miss.belief = belief_miss.belief / s if s > 1e-12 else np.ones(64) / 64.0
                    belief_miss.predict()
                    eval_val, _ = minimax(child, belief_miss, cell_potential, depth - 1, alpha, beta, not is_max, deadline, counter)
                    
                    ev = 4.0 * p_find - 2.0 * (1.0 - p_find)
                    eval_val += (ev * SD_WEIGHT)
            else:
                new_belief = belief.copy()
                new_belief.predict()
                eval_val, _ = minimax(child, new_belief, cell_potential, depth - 1, alpha, beta, not is_max, deadline, counter)

            if eval_val > max_eval:
                max_eval = eval_val
                best_move = move
            alpha = max(alpha, eval_val)
            if beta <= alpha: break
        return max_eval, best_move

    else:
        min_eval = float('inf')
        for move in ordered_moves:
            child = board.forecast_move(move)
            if child is None: continue
            child.reverse_perspective()

            if move.move_type.name == 'SEARCH':
                bx, by = belief.best_search_target()
                target_idx = by * 8 + bx
                p_find = belief.belief[target_idx]

                new_belief = belief.copy()
                new_belief.predict()
                eval_val, _ = minimax(child, new_belief, cell_potential, depth - 1, alpha, beta, not is_max, deadline, counter)
                
                # Discounted Opponent EV (Albert might not be perfectly tracking the rat)
                ev = 4.0 * p_find - 2.0 * (1.0 - p_find)
                if ev > 0:
                    eval_val -= (ev * 0.5 * SD_WEIGHT) 
            else:
                new_belief = belief.copy()
                new_belief.predict()
                eval_val, _ = minimax(child, new_belief, cell_potential, depth - 1, alpha, beta, not is_max, deadline, counter)

            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha: break
        return min_eval, best_move

class SearchEngine:
    def run(self, board, belief, budget_seconds, cell_potential):
        deadline = time.time() + budget_seconds - 0.35 
        
        moves = all_legal_moves(board, belief)
        if not moves: raise RuntimeError("No legal moves!")
        best_move = order_moves(moves, belief)[0] 
        
        depth = 1
        counter = [0]
        while True:
            try:
                val, move = minimax(board, belief, cell_potential, depth, float('-inf'), float('inf'), True, deadline, counter)
                best_move = move
                depth += 1
            except TimeoutError:
                break
                
        return best_move
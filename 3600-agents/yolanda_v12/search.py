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
    if counter[0] % 500 == 0:
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
    
    best_eval = float('-inf') if is_max else float('inf')
    
    for move in ordered_moves:
        child = board.forecast_move(move)
        if child is None: continue
        child.reverse_perspective()

        # EXPECTIMINIMAX CHANCE NODE BRANCHING
        if move.move_type.name == 'SEARCH':
            bx, by = belief.best_search_target()
            target_idx = by * 8 + bx
            p_hit = belief.belief[target_idx]
            
            sd_weight = WEIGHTS['sd'] # Grab your current tuned score multiplier
            
            # Only split the universe if there is a realistic chance of hitting
            if p_hit > 0.05:
                # UNIVERSE 1: THE MISS (Probability: 1 - p_hit)
                miss_belief = belief.copy()
                miss_belief.belief[target_idx] = 0.0
                s = miss_belief.belief.sum()
                miss_belief.belief = miss_belief.belief / s if s > 1e-12 else np.ones(64) / 64.0
                miss_belief.predict()
                
                eval_miss, _ = minimax(child, miss_belief, cell_potential, depth - 1, alpha, beta, not is_max, deadline, counter)
                
                # Apply penalty mathematically (-2 points)
                if is_max: eval_miss -= (2.0 * sd_weight)
                else: eval_miss += (2.0 * sd_weight)

                # UNIVERSE 2: THE HIT (Probability: p_hit)
                hit_belief = belief.copy()
                hit_belief.initialize() # Rat resets
                hit_belief.predict()
                
                eval_hit, _ = minimax(child, hit_belief, cell_potential, depth - 1, alpha, beta, not is_max, deadline, counter)
                
                # Apply reward mathematically (+4 points)
                if is_max: eval_hit += (4.0 * sd_weight)
                else: eval_hit -= (4.0 * sd_weight)
                
                # Mathematical Expectation: E[V] = P(hit)*V(hit) + P(miss)*V(miss)
                eval_val = (p_hit * eval_hit) + ((1.0 - p_hit) * eval_miss)
            else:
                # Treat as guaranteed miss to save depth computation
                new_belief = belief.copy()
                new_belief.belief[target_idx] = 0.0
                s = new_belief.belief.sum()
                new_belief.belief = new_belief.belief / s if s > 1e-12 else np.ones(64) / 64.0
                new_belief.predict()
                
                eval_val, _ = minimax(child, new_belief, cell_potential, depth - 1, alpha, beta, not is_max, deadline, counter)
                
                # Apply the guaranteed miss penalty mathematically (-2 points)
                if is_max: eval_val -= (2.0 * sd_weight)
                else: eval_val += (2.0 * sd_weight)
                
        else:
            # DETERMINISTIC NODE
            new_belief = belief.copy()
            new_belief.predict()
            eval_val, _ = minimax(child, new_belief, cell_potential, depth - 1, alpha, beta, not is_max, deadline, counter)
            
        if is_max:
            if eval_val > best_eval:
                best_eval = eval_val
                best_move = move
            alpha = max(alpha, eval_val)
        else:
            if eval_val < best_eval:
                best_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            
        if beta <= alpha:
            break
            
    return best_eval, best_move

class SearchEngine:
    def run(self, board, belief, budget_seconds, cell_potential):
        from .heuristic import _eval_cache
        _eval_cache.clear()
        
        deadline = time.time() + budget_seconds - 0.35 
        
        moves = all_legal_moves(board, belief)
        if not moves: raise RuntimeError("No legal moves!")
        best_move = order_moves(moves, belief)[0] 
        
        depth = 1
        counter = [0]
        while True:
            try:
                val, move = minimax(board, belief, cell_potential, depth, float('-inf'), float('inf'), True, deadline, counter)
                if move is not None:
                    best_move = move
                depth += 1
            except TimeoutError:
                break
                
        return best_move
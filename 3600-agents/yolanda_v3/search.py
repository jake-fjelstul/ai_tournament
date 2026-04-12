import time
from .heuristic import evaluate, CARPET_SCORE
from .move_gen import all_legal_moves
try:
    from game.enums import Result
except ImportError:
    from engine.game.enums import Result

def order_moves(moves, belief, is_max=True):
    def quick_score(move):
        if hasattr(move, 'roll_length') and move.roll_length:
            #if move.roll_length == 1:
            #   return -10.0
            return CARPET_SCORE.get(move.roll_length, 0) * 3
            
        if move.move_type.name == 'SEARCH': 
            return belief.search_ev() * 2  # Scaled back from 5
            
        if move.move_type.name == 'PRIME': 
            return 1.0
            
        return 0.0 # PlainStep
        
    return sorted(moves, key=quick_score, reverse=True)

def minimax(board, belief, cell_potential, depth, alpha, beta, is_max, deadline):
    if time.time() >= deadline:
        raise TimeoutError()
        
    if board.is_game_over():
        winner = board.get_winner()
        if winner == Result.TIE:
            return 0, None
        elif winner == Result.PLAYER:
            return (float('inf') if is_max else float('-inf')), None
        elif winner == Result.ENEMY:
            return (float('-inf') if is_max else float('inf')), None
        else:
            return evaluate(board, belief, cell_potential, is_max), None

    if depth == 0:
        return evaluate(board, belief, cell_potential, is_max), None

    moves = all_legal_moves(board, belief)
    if not moves:
        return evaluate(board, belief, cell_potential, is_max), None

    ordered_moves = order_moves(moves, belief, is_max)
    best_move = ordered_moves[0]
    
    if is_max:
        max_eval = float('-inf')
        for move in ordered_moves:
            child = board.forecast_move(move)
            if child is None: continue
            child.reverse_perspective()
            
            new_belief = belief.copy()
            new_belief.predict()
            
            eval_val, _ = minimax(child, new_belief, cell_potential, depth - 1, alpha, beta, False, deadline)
            if move.move_type.name == 'SEARCH':
                eval_val += belief.search_ev()
            
            if eval_val > max_eval:
                max_eval = eval_val
                best_move = move
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            child = board.forecast_move(move)
            if child is None: continue
            child.reverse_perspective()
            
            new_belief = belief.copy()
            new_belief.predict()
            
            eval_val, _ = minimax(child, new_belief, cell_potential, depth - 1, alpha, beta, True, deadline)
            if move.move_type.name == 'SEARCH':
                eval_val -= belief.search_ev()
                
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        return min_eval, best_move

class SearchEngine:
    def run(self, board, belief, budget_seconds, cell_potential):
        deadline = time.time() + budget_seconds - 0.35 # 350ms safety margin
        
        moves = all_legal_moves(board, belief)
        if not moves: raise RuntimeError("No legal moves!")
        best_move = order_moves(moves, belief)[0] 
        
        depth = 1
        while True:
            try:
                val, move = minimax(board, belief, cell_potential, depth, float('-inf'), float('inf'), True, deadline)
                best_move = move
                depth += 1
            except TimeoutError:
                break
                
        return best_move

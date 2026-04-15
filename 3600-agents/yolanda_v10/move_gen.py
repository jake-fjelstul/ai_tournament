from typing import List
try:
    from game.board import Board
    from game.enums import Direction, MoveType, BOARD_SIZE
    from game.move import Move
except ImportError:
    from engine.game.board import Board
    from engine.game.enums import Direction, MoveType, BOARD_SIZE
    from engine.game.move import Move
from .rat_belief import RatBelief


DIR_MAP = {
    (0, -1): Direction.UP,
    (0,  1): Direction.DOWN,
    (-1,  0): Direction.LEFT,
    (1,  0): Direction.RIGHT
}

def plain_steps(board: Board) -> List[Move]:
    x, y = board.player_worker.position
    opp = board.opponent_worker.position
    moves = []
    
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
            continue
        
        bit = 1 << (ny * BOARD_SIZE + nx)
        if board._blocked_mask & bit: continue
        if board._primed_mask & bit: continue
        if (nx, ny) == opp: continue
        
        moves.append(Move.plain(DIR_MAP[(dx, dy)]))
    return moves

def prime_steps(board: Board) -> List[Move]:
    x, y = board.player_worker.position
    opp = board.opponent_worker.position
    
    current_bit = 1 << (y * BOARD_SIZE + x)
    if (board._primed_mask | board._carpet_mask) & current_bit:
        return []

    moves = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
            continue
            
        bit = 1 << (ny * BOARD_SIZE + nx)
        if board._blocked_mask & bit: continue
        if board._primed_mask & bit: continue # Cannot land on a primed square
        if (nx, ny) == opp: continue
        
        moves.append(Move.prime(DIR_MAP[(dx, dy)]))
    return moves

def carpet_rolls(board: Board) -> List[Move]:
    x, y = board.player_worker.position
    opp = board.opponent_worker.position
    moves = []
    
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        roll_length = 0
        cx, cy = x + dx, y + dy
        
        while (0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE):
            if (cx, cy) == opp:
                break
            
            bit = 1 << (cy * BOARD_SIZE + cx)
            if not (board._primed_mask & bit):
                break
            
            roll_length += 1
            moves.append(Move.carpet(DIR_MAP[(dx, dy)], roll_length))
            
            cx += dx
            cy += dy

    return moves

def search_moves(belief: RatBelief = None) -> List[Move]:
    if belief is None:
        return []
    if belief.search_ev() < 0.5: 
        return []
    bx, by = belief.best_search_target()
    return [Move.search((bx, by))]

def all_legal_moves(board, belief=None):
    all_carpets = carpet_rolls(board)
    good_carpets = [m for m in all_carpets if m.roll_length > 1]
    spatial_moves = (good_carpets or all_carpets) + prime_steps(board) + plain_steps(board)

    if not spatial_moves:
        bx, by = belief.best_search_target() if belief else (0, 0)
        return [Move.search((bx, by))]

    if belief is not None and belief.search_ev() > 0.0:
        bx, by = belief.best_search_target()
        spatial_moves = spatial_moves + [Move.search((bx, by))]

    return spatial_moves

import chess
import numpy as np


def get_board_state(board: chess.Board):
    """
    get board state representation from the pov of current player (board.turn).

    state vector description:
    0-63 - board pieces from my pov (a1-h8 for white, reversed for black)
    64-71 - squares my pawns can en passant to
    72-73 - my castling rights (left, right)
    74-75 - opponent's castling rights (left, right)

    vocabulary description:
    0 - empty space
    1-6 - my pieces
    7-12 - opponent's pieces
    13 - en passant unavailable
    14 - en passant available
    15 - castle unavailable
    16 - castle available
    """
    pieces = [0] * 64
    for square, piece in board.piece_map().items():
        pieces[square] = piece.piece_type + (piece.color ^ board.turn) * 6
    ep = [13] * 8
    if board.ep_square:
        ep[board.ep_square % 8] = 14
    castle = [16 if board.castling_rights & c else 15
              for c in [chess.BB_A1, chess.BB_H1, chess.BB_A8, chess.BB_H8]]
    if board.turn == chess.BLACK:
        pieces = pieces[::-1]
        ep = ep[::-1]
        castle = castle[::-1]
    state = pieces + ep + castle
    return state


MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT,
    UP2_RIGHT, UP_LEFT2, UP_RIGHT2, DOWN2_LEFT, DOWN2_RIGHT, DOWN_LEFT2,
    DOWN_RIGHT2, UP_LEFT_PROMOTE_QUEEN, UP_LEFT_PROMOTE_ROOK,
    UP_LEFT_PROMOTE_KNIGHT, UP_LEFT_PROMOTE_BISHOP, UP_PROMOTE_QUEEN,
    UP_PROMOTE_ROOK, UP_PROMOTE_KNIGHT, UP_PROMOTE_BISHOP,
    UP_RIGHT_PROMOTE_QUEEN, UP_RIGHT_PROMOTE_ROOK, UP_RIGHT_PROMOTE_KNIGHT,
    UP_RIGHT_PROMOTE_BISHOP
] = range(28)


def get_move(move: chess.Move, color: chess.Color):
    """
    move vocab description:

    see MOVE_DIRECTION constants
    """
    from_sq = move.from_square
    to_sq = move.to_square
    if color == chess.BLACK:
        to_sq = 64 - to_sq - 1
        from_sq = 64 - from_sq - 1
    from_rank, from_file = divmod(from_sq, 8)
    to_rank, to_file = divmod(to_sq, 8)
    dir_file = to_file - from_file
    dir_rank = to_rank - from_rank
    if move.promotion is not None:
        if move.promotion == chess.QUEEN:
            move_direction = UP_PROMOTE_QUEEN
        elif move.promotion == chess.ROOK:
            move_direction = UP_PROMOTE_ROOK
        elif move.promotion == chess.KNIGHT:
            move_direction = UP_PROMOTE_KNIGHT
        elif move.promotion == chess.BISHOP:
            move_direction = UP_PROMOTE_BISHOP
        else:
            raise RuntimeError
        move_direction += 4 * dir_file
    elif dir_rank == 2 and dir_file == -1:
        move_direction = UP2_LEFT
    elif dir_rank == 2 and dir_file == 1:
        move_direction = UP2_RIGHT
    elif dir_rank == -2 and dir_file == -1:
        move_direction = DOWN2_LEFT
    elif dir_rank == -2 and dir_file == 1:
        move_direction = DOWN2_RIGHT
    elif dir_rank == 1 and dir_file == -2:
        move_direction = UP_LEFT2
    elif dir_rank == 1 and dir_file == 2:
        move_direction = UP_RIGHT2
    elif dir_rank == -1 and dir_file == -2:
        move_direction = DOWN_LEFT2
    elif dir_rank == -1 and dir_file == 2:
        move_direction = DOWN_RIGHT2
    elif dir_rank > 0 and dir_file == 0:
        move_direction = UP
    elif dir_rank > 0 and dir_file < 0:
        move_direction = UP_LEFT
    elif dir_rank > 0 and dir_file > 0:
        move_direction = UP_RIGHT
    elif dir_rank == 0 and dir_file < 0:
        move_direction = LEFT
    elif dir_rank == 0 and dir_file > 0:
        move_direction = RIGHT
    elif dir_rank < 0 and dir_file == 0:
        move_direction = DOWN
    elif dir_rank < 0 and dir_file < 0:
        move_direction = DOWN_LEFT
    elif dir_rank < 0 and dir_file > 0:
        move_direction = DOWN_RIGHT
    else:
        raise RuntimeError
    return move_direction * 64 + to_sq


def get_value(comment: str, color: chess.Color):
    try:
        value = float(comment.split('/')[0])
        if color == chess.BLACK:
            value *= -1
        return 1 / (1 + np.exp(-value * 0.5))
    except ValueError:
        return 0.5


def get_result(result: str, color: chess.Color):
    if result == '1-0' and color == chess.WHITE:
        return 1.
    elif result == '0-1' and color == chess.BLACK:
        return 1.
    elif result == '1-0' and color == chess.BLACK:
        return 0.
    elif result == '0-1' and color == chess.WHITE:
        return 0.
    else:
        return 0.5

import os
import argparse
import glob
import io
import random

import chess
import chess.pgn
import numpy as np
import h5py

from .helpers import get_board_state, get_move, get_value, get_result


def get_policy_value_labels(pgn_str: str):
    """
    get policy/value labels from all positions in a game.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_str))
    rows = []
    result_header = game.headers.get('Result', None)
    board = chess.Board()
    for node in game.mainline():
        next_node = node.next()
        if not next_node:
            continue
        try:
            board.push(node.move)
        except (AssertionError, ValueError) as e:
            board = node.board()
        state = get_board_state(board)
        move = get_move(next_node.move, board.turn)
        value = get_value(node.comment, board.turn) if node.comment else 0.5
        result = get_result(result_header, board.turn) if result_header else 0.5
        rows.append((*state, move, value, result))
    return rows


def main(pgn_dir, output_dir, n_parts=5):
    print('Loading pgns')
    pgn_paths = glob.glob(os.path.join(pgn_dir, '*.pgn'))
    games = []
    for pgn_path in pgn_paths:
        with open(pgn_path, 'r') as pgn:
            pgn_str = ''
            while True:
                try:
                    line = pgn.readline()
                except UnicodeDecodeError as e:
                    print(e)
                    continue
                if not line:
                    break
                if not line.strip() and not pgn_str.strip().endswith(']'):
                    games.append(pgn_str)
                    pgn_str = ''
                else:
                    pgn_str += line
    print(f'Loaded {len(games)} games')
    random.shuffle(games)
    parts = [games[i::n_parts] for i in range(n_parts)]
    total_length = 0
    for i, part in enumerate(parts):
        print(f'Processing part {i+1}/{len(parts)}')
        rows = []
        for j, game in enumerate(part):
            rows.extend(get_policy_value_labels(game))
            if j % 10000 == 0:
                print(f'{j}/{len(part)} games')
        random.shuffle(rows)
        rows = np.array(rows,
            dtype=[*[(f'state_{i}', 'u2') for i in range(len(rows[0]) - 3)],
                   ('move', 'u2'), ('value', 'f4'), ('result', 'f2')])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        h5_path = os.path.join(output_dir, f'{i}.h5')
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('policy_value_labels', data=rows)
        print(f'Saved {len(rows)} positions to {h5_path}')
        total_length += len(rows)
    print(f'Saved {total_length} positions to {n_parts} parts in {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pgn_dir',
                        help='Path to directory of pgns',
                        required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Path to output directory',
                        required=True)
    args = parser.parse_args()
    main(args.pgn_dir, args.output_dir)

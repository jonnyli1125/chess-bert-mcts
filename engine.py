import argparse

import chess

from mcts import MCTSAgent


class BertMCTSPlayer:
    def __init__(self, ckpt_path):
        self.mcts = MCTSAgent(ckpt_path)

    def uci(self):
        print('id name chess-bert-mcts')
        print('uciok')

    def ucinewgame(self):
        pass

    def setoption(self, option):
        pass

    def quit(self):
        self.mcts.quit()

    def isready(self):
        print('readyok')

    def position(self, moves):
        if moves[0] == 'fen':
            self.mcts.set_position(' '.join(moves[1:]))
        elif moves[0] == 'startpos':
            self.mcts.set_position(moves[2:])

    def go(self):
        if self.mcts.board.is_game_over():
            return
        bestmove, info = self.mcts.get_bestmove_and_info()
        if info:
            print('info', *[f'{k} {v}' for k, v in info.items()])
        print('bestmove', bestmove.uci())


def main(args):
    player = BertMCTSPlayer(args.ckpt_path)
    print('chess-bert-mcts running')
    while True:
        cmd_line = input()
        cmd = cmd_line.split(' ', 1)
        cmd = [c.rstrip() for c in cmd]

        if cmd[0] == 'uci':
            player.uci()
        elif cmd[0] == 'setoption':
            option = cmd[1].split(' ')
            player.setoption(option)
        elif cmd[0] == 'isready':
            player.isready()
        elif cmd[0] == 'ucinewgame':
            player.ucinewgame()
        elif cmd[0] == 'position':
            moves = cmd[1].split(' ')
            player.position(moves)
        elif cmd[0] == 'go':
            player.go()
        elif cmd[0] == 'quit':
            player.quit()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--ckpt_path', help='Path to model checkpoint',
                        required=True)
    args = parser.parse_args()
    main(args)

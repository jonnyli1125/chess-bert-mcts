import time

import chess
import chess.polyglot
import numpy as np
from scipy.special import softmax
import torch

from model import BertPolicyValue
from utils.helpers import get_board_state, get_move


np.seterr(divide='ignore')


class Node:
    def __init__(self, board: chess.Board):
        self.moves = list(board.legal_moves)  # [chess.Move]
        self.children = [None for _ in self.moves]  # [Node]
        self.policy = None  # [float] or None
        self.value = None  # float or None
        self.count = 1  # int


class MCTSAgent:
    def __init__(self, ckpt_path, n_playouts=1000, c_puct=1):
        self.model = BertPolicyValue.load_from_checkpoint(ckpt_path)
        self.model.cuda()
        self.model.eval()
        self.n_playouts = n_playouts
        self.c_puct = c_puct
        self.node_lookup = {}  # zobrist hash -> Node
        self.board = chess.Board()  # use global board reference to save memory

    def set_position(self, fen_or_moves):
        if isinstance(fen_or_moves, str):
            self.board.set_fen(fen_or_moves)
        elif isinstance(fen_or_moves, list):
            self.board.reset()
            for move in fen_or_moves:
                self.board.push_uci(move)

    def get_bestmove_and_info(self):
        """get best move and info dict given the board position."""
        # initialize
        self.node_lookup.clear()
        root = self.expand_and_evaluate_node()
        # return early if only 1 legal move
        if len(root.moves) == 1:
            return root.moves[0], {}
        # execute playouts
        start_time = time.time()
        for _ in range(self.n_playouts):
            self.search(root)
        total_time = time.time() - start_time
        # get best move (child node with highest visit count)
        i = np.argmax([c.count if c else 0 for c in root.children])
        bestmove = root.moves[i]
        bestmove_value = root.children[i].value / root.children[i].count
        if bestmove_value == 1:
            cp = 4000
        elif bestmove_value == 0:
            cp = -4000
        else:
            cp = int(np.arctanh(bestmove_value) * 400)
        info = {
            'cp': cp,
            'time': int(total_time * 1000),
            'nodes': len(self.node_lookup),
            'nps': int(len(self.node_lookup) / total_time),
            'pv': bestmove.uci()
        }
        return bestmove, info

    def search(self, node: Node, invert_child_value=True):
        """execute one iteration of MCTS"""
        # select max ucb child
        i = self.select_max_ucb_child(node)
        self.board.push(node.moves[i])
        # if child has not been visited before, expand and evaluate
        if node.children[i] is None:
            node.children[i] = self.expand_and_evaluate_node(invert_child_value)
            child_value = node.children[i].value
        # otherwise recurse on child
        else:
            child_value = self.search(node.children[i], not invert_child_value)
        # backpropagate
        node.value += child_value
        node.count += 1
        self.board.pop()
        return child_value

    def select_max_ucb_child(self, node: Node):
        """get index of child node that maximizes ucb value"""
        Q = np.repeat(0.5, len(node.children))
        P = node.policy
        N = np.zeros(len(node.children))
        children_i = [i for i, c in enumerate(node.children) if c is not None]
        for i, child in enumerate(node.children):
            if child is not None:
                Q[i] = child.value / child.count
                N[i] = child.count
        U = Q + self.c_puct * P * (np.sqrt(node.count) / (1 + N))
        return np.argmax(U)

    def expand_and_evaluate_node(self, invert_value=False):
        """create and evaluate node of current board position"""
        z_hash = chess.polyglot.zobrist_hash(self.board)
        # check if node has already been reached
        if z_hash in self.node_lookup:
            return self.node_lookup[z_hash]
        # otherwise create new node
        node = Node(self.board)
        self.node_lookup[z_hash] = node
        # evaluate node
        if self.board.is_checkmate():
            node.value = -1
        elif self.board.is_game_over():
            node.value = 0
        else:
            # TODO evaluate nodes in batches by layer on tree
            state = get_board_state(self.board)
            input_ids = torch.tensor([state], dtype=torch.long).cuda()
            with torch.no_grad():
                output = self.model(input_ids)
                value = output['value'].detach().cpu().numpy()[0]
                policy_logits = output['policy'].detach().cpu().numpy()[0]
            legal_moves = [get_move(m, self.board.turn) for m in node.moves]
            policy = softmax(policy_logits[legal_moves])
            node.value = value
            node.policy = policy
        if invert_value:
            node.value *= -1
        return node

import time
from threading import Thread, Lock

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
        self.policy = np.ones(len(self.moves)) / len(self.moves)  # [float]
        self.value = 0  # float
        self.count = 0  # int
        self.lock = Lock()


class MCTSAgent:
    def __init__(self, ckpt_path, n_playouts=800, c_puct=1, batch_size=8):
        self.model = BertPolicyValue.load_from_checkpoint(ckpt_path)
        self.model.cuda()
        self.model.eval()
        self.n_playouts = n_playouts
        self.c_puct = c_puct
        self.node_lookup = {}  # zobrist hash -> Node
        self.node_lookup_lock = Lock()
        self.board = chess.Board()  # use global board reference to save memory
        self.batch_size = batch_size

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
        root = self.expand_node()
        self.evaluate_node_batch([[root]], [self.board])
        # return early if only 1 legal move
        if len(root.moves) == 1:
            return root.moves[0], {}
        # execute playouts
        start_time = time.time()
        threads = []
        for i in range(self.n_playouts // self.batch_size):
            # select leaf nodes and apply virtual losses
            paths, boards = zip(*(self.select_leaf_node(root)
                                  for j in range(self.batch_size)))
            # evaluate batch and backpropagate
            t = Thread(target=self.evaluate_node_batch, args=(paths, boards))
            threads.append(t)
            t.start()
            #self.evaluate_node_batch(paths, boards)
        for t in threads:
            t.join()
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

    def select_leaf_node(self, node: Node):
        """
        traverse to leaf node and expand tree.
        returns path from root to leaf node and board state of leaf node.
        """
        path = [node]
        # apply virtual loss
        node.lock.acquire()
        node.value -= 1
        node.count += 1
        node.lock.release()
        # terminal state reached
        if self.board.is_game_over():
            return path, self.board.copy()
        # select max ucb child
        node.lock.acquire()
        i = self.select_max_ucb_child(node)
        child = node.children[i]
        self.board.push(node.moves[i])
        node.lock.release()
        # if child has not been visited before, expand and apply virtual loss
        if child is None:
            leaf_node = self.expand_node()
            node.lock.acquire()
            node.children[i] = leaf_node
            node.lock.release()
            path.append(leaf_node)
            leaf_board = self.board.copy()
        # otherwise recurse on child
        else:
            child_path, leaf_board = self.select_leaf_node(child)
            path += child_path
        self.board.pop()
        return path, leaf_board

    def select_max_ucb_child(self, node: Node):
        """get index of child node that maximizes ucb value"""
        n = len(node.children)
        Q = np.zeros(n)
        P = node.policy
        C = np.zeros(n)
        for i, child in enumerate(node.children):
            if child is not None:
                Q[i] = child.value / child.count
                C[i] = child.count
        U = Q + self.c_puct * P * (np.sqrt(node.count) / (1 + C))
        return np.argmax(U)

    def expand_node(self):
        """create node of current board position and apply virtual loss"""
        z_hash = chess.polyglot.zobrist_hash(self.board)
        # check if node has already been reached
        self.node_lookup_lock.acquire()
        if z_hash in self.node_lookup:
            node = self.node_lookup[z_hash]
        # otherwise create new node
        else:
            node = Node(self.board)
            self.node_lookup[z_hash] = node
        self.node_lookup_lock.release()
        # virtual loss
        node.lock.acquire()
        node.value -= 1
        node.count += 1
        node.lock.release()
        return node

    def evaluate_node_batch(self, paths, boards):
        """
        evaluate batch of nodes with model and backpropagate

        - nodes are taken from last element of each path
        - paths are required for backpropagation
        - boards are required for evaluation
        """
        states = [get_board_state(board) for board in boards]
        policies, values = self.compute_policy_value_batch(states)
        for i, x in enumerate(zip(paths, boards, policies, values)):
            path, board, policy, value = x
            node = path[-1]
            # filter policy by legal moves
            legal_moves = [get_move(m, board.turn) for m in node.moves]
            policy = softmax(policy[legal_moves]) if legal_moves else []
            # check node for terminal states
            if board.is_checkmate():
                value = -1
            elif board.is_game_over():
                value = 0
            # apply policy and backpropagate value
            node.lock.acquire()
            node.policy = policy
            node.lock.release()
            for j, path_node in enumerate(path):
                invert = 1 if j % 2 == 0 else -1
                path_node.lock.acquire()
                path_node.value += invert * value + 1
                path_node.lock.release()

    def compute_policy_value_batch(self, states):
        """evaluate batch of board states on gpu"""
        input_ids = torch.tensor(states, dtype=torch.long).cuda()
        with torch.no_grad():
            output = self.model(input_ids)
            policy = output['policy'].detach().cpu().numpy()
            value = output['value'].detach().cpu().numpy()
        return policy, value

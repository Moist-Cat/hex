from gemu import HexBoard
from dsa import minimax, distance_heuristic, average_distance_heuristic

import unittest


def place(matrix, hex_board):
    for r, column in enumerate(matrix):
        for c, piece in enumerate(column):
            res = hex_board.place_piece(r, c, piece)
            if not res:
                raise Exception("Placing pieces in a dirty board!")


class TestAux(unittest.TestCase):
    def setUp(self):
        self.board = HexBoard(5)

    def test_conn_basic(self):
        self.assertFalse(self.board.check_connection(1))

        mat = [
             [1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
        ]

        place(mat, self.board)
        self.assertTrue(self.board.check_connection(1))

    def test_conn_tiny_diagonal(self):
        mat = [
              [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 2],
              [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 2],
              [0, 0, 0, 2, 0],
        ]

        place(mat, self.board)
        self.assertTrue(self.board.check_connection(2))

    def test_conn_blocked(self):
        mat = [
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 1],
        ]

        place(mat, self.board)
        self.assertFalse(self.board.check_connection(1))

    def test_wrong_side(self):
        mat = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2],
        ]

        place(mat, self.board)
        self.assertFalse(self.board.check_connection(1))

    def test_conn_genuine_diagonal_again(self):
        mat = [
              [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
              [1, 0, 0, 0, 0],
        ]

        place(mat, self.board)
        self.assertTrue(self.board.check_connection(1))

    def test_fake_diagonal(self):
        mat = [
              [1, 0, 0, 0, 2],
            [0, 1, 0, 2, 0],
              [0, 0, 1, 0, 0],
            [0, 2, 0, 1, 0],
              [2, 0, 0, 0, 1],
        ]

        place(mat, self.board)
        self.assertFalse(self.board.check_connection(1))

    def test_clone_operation(self):
        """Test board cloning"""
        mat = [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 1],
        ]
        place(mat, self.board)
        cloned = self.board.clone()

        for r in range(self.board.size):
            for c in range(self.board.size):
                self.assertEqual(self.board.board[r][c], cloned.board[r][c])

        cloned._real_clone()
        cloned.place_piece(0, 1, 1)
        cloned._sync()
        self.assertEqual(self.board.board[0][1], 0)
        self.assertEqual(cloned.board[0][1], 1)

class TestMinimax(unittest.TestCase):

    def test_forced_win(self):
        board = HexBoard(3)
        # Setup board for AI (player 2) to win with (2,2)
        board.place_piece(0, 0, 2)
        board.place_piece(1, 1, 2)
        
        def heuristic(b, pid):
            return 100 if b.check_connection(2) else 0
        
        val, move = minimax(
            board, 
            depth=2,
            alpha=float("-inf"),
            beta=float("inf"),
            maximising=True,
            player_id=2,
            heuristic=heuristic
        )
        self.assertEqual(move, (2, 0))  # Winning move
        board.place_piece(*move, 2)
        self.assertTrue(board.check_connection(2))
        self.assertEqual(val, 100)

    def test_blocking_move(self):
        board = HexBoard(3)
        board.place_piece(1, 0, 1)
        board.place_piece(1, 2, 1)
        
        def heuristic(b, pid):
            return -100 if b.check_connection(1) else 0
        
        val, move = minimax(
            board,
            depth=2,
            alpha=float("-inf"),
            beta=float("inf"),
            maximising=True,
            player_id=2,
            heuristic=heuristic
        )
        self.assertEqual(move, (1, 1)) # blocked!

    def test_heuristic_propagation(self):
        board = HexBoard(2)  # 2x2 board
        heuristic_values = {
            (0,0): 5,
            (0,1): 3,
            (1,0): -2,
            (1,1): 7
        }
        
        def mock_heuristic(b, pid):
            moves = b.player_positions[3 - pid].intersection(set(heuristic_values.keys()))
            return -sum((heuristic_values.get(move, -99) for move in moves))

        val, move = minimax(
            board,
            depth=1,
            alpha=-69,
            beta=float("inf"),
            maximising=True,
            player_id=2,
            heuristic=mock_heuristic
        )
        # 2 maximises then 1 minimizes; -2, since is adventageous to player 2
        # it's disadventageous to player 1 so -2 becomes 2 and so on.
        # Then 2 chooses the max between -5, -3, 2, 7
        self.assertEqual(val, 2, val)

    def test_alpha_beta_pruning(self):
        board = HexBoard(3)
        visited_nodes = []
        
        def tracing_heuristic(b, pid):
            visited_nodes.append(tuple(sorted(b.player_positions[pid])))
            return len(b.player_positions[pid])
        
        minimax(
            board,
            depth=2,
            alpha=3,  # Set high alpha
            beta=float("inf"),
            maximising=True,
            player_id=2,
            heuristic=tracing_heuristic
        )
        
        # Should prune branches once a value >= 3 is found
        self.assertEqual(len(visited_nodes), 9) # all possible responses to the first moves

    def test_depth_limitation(self):
        board = HexBoard(3)
        board.place_piece(0, 0, 2)
        
        depth = 2
        def depth_check_heuristic(b, pid):
            assert len(b.player_positions[2]) <= depth + 1
            return 0
        
        minimax(
            board,
            depth=depth,
            alpha=-9999,
            beta=9999,
            maximising=True,
            player_id=2,
            heuristic=depth_check_heuristic
        )

class TestHeuristic(unittest.TestCase):


    def test_immediate_win(self):
        board = HexBoard(3)
        # Player 1 already has left-right connection
        board.place_piece(0, 0, 1)
        board.place_piece(0, 1, 1)
        board.place_piece(0, 2, 1)
        
        h = distance_heuristic(board, 1)
        assert abs(h) == 0  # No pieces needed to win

    def test_completely_blocked(self):
        board = HexBoard(3)
        # Player 2 completely blocks left-right
        board.place_piece(0, 1, 2)
        board.place_piece(1, 1, 2)
        board.place_piece(2, 1, 2)
        
        h = distance_heuristic(board, 1)
        assert abs(h) == float('inf')  # Impossible to win

    def test_minimum_path_pieces(self):
        board = HexBoard(3)
        # Player 1 needs 2 more pieces to connect
        board.place_piece(0, 0, 1)
        board.place_piece(2, 2, 1)
        
        h = distance_heuristic(board, 1)
        assert abs(h) == 2  # Needs (1,1) and either (0,1) or (1,0)

    def test_island_distances(self):
        board = HexBoard(5)
        # Player 2 has two islands separated by 2 empty spaces
        board.place_piece(0, 0, 2)
        board.place_piece(0, 1, 2)
        board.place_piece(4, 3, 2)
        board.place_piece(4, 4, 2)

        h = distance_heuristic(board, 2)
        h_2 = average_distance_heuristic(board, 2)
        assert abs(h) == 3
        assert abs(h_2) == 3, h_2 # two islands


    def test_phantom_edge_connection(self):
        board = HexBoard(3)
        # Player 1 has piece touching left edge
        board.place_piece(1, 0, 1)
        
        h = distance_heuristic(board, 1)
        assert abs(h) == 2, h  # Needs to connect to right edge

    def test_opponent_blocks(self):
        board = HexBoard(3)
        board.place_piece(2, 0, 1)
        board.place_piece(2, 2, 1)

        h = distance_heuristic(board, 1)
        assert abs(h) == 1  # direct in 2, 1

        board.place_piece(2, 1, 2)  # blocked!

        h = distance_heuristic(board, 1)

        assert abs(h) == 2 # must go around

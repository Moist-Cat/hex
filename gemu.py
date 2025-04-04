from collections import deque
import math
import random

from dsa import (
    HexProblem,
    astar_search,
    failure,
    cutoff,
    path_actions,
    get_neighbors,
    minimax,
    nun,
    manhattan,
)

# abstract
class HexBoard:
    def __init__(self, size: int):
        self.size = size
        self.board = [[0] * size for _ in range(size)]  # 0=empty, 1=Player1, 2=Player2
        self.player_positions = {1: set(), 2: set()}

    def clone(self) -> "HexBoard":
        new_board = HexBoard(self.size)
        # new_board.board = [row[:] for row in self.board]
        new_board.board = self.board
        new_board.player_positions = {
            1: self.player_positions[1].copy(),
            2: self.player_positions[2].copy(),
        }
        return new_board

    def wipe(self):
        for r, column in enumerate(self.board):
            for c, _ in enumerate(column):
                self.place_piece(r, c, 0)

    def _sync(self):
        for r in range(self.size):
            for c in range(self.size):
                self.board[r][c] = 0

        for r, c in self.player_positions[1]:
            self.board[r][c] = 1
        for r, c in self.player_positions[2]:
            self.board[r][c] = 2

    def _real_clone(self):
        self.board = [row[:] for row in self.board]

    def _unused_tile(self, row, col):
        return not (
            (row, col) in self.player_positions[1]
            or (row, col) in self.player_positions[2]
        )

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        if player_id == 0:
            return True

        if not self._unused_tile(row, col):
            return False

        self.player_positions[player_id].add((row, col))

        return True

    def get_possible_moves(self) -> list:
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self._unused_tile(r, c)
        ]

    def check_connection(self, player_id: int) -> bool:
        if len(self.player_positions[player_id]) < self.size:
            return False

        if player_id == 2:
            # up
            start_nodes = [
                (x, y) for x, y in self.player_positions[player_id] if x == 0
            ]
        elif player_id == 1:
            # left
            start_nodes = [
                (x, y) for x, y in self.player_positions[player_id] if y == 0
            ]
        else:
            print("Invalid player", player_id)
            return False
        target = self.size - 1

        visited = set()
        queue = start_nodes

        # O(self.player_positions[player_id]) -> O(n**2)
        while queue:
            r, c = queue.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))

            # short circuit if it's the wrong player
            # avoids NameError
            if (player_id == 2 and r == target) or (player_id == 1 and c == target):
                return True

            for coords in get_neighbors(r, c):
                if coords in self.player_positions[player_id]:
                    queue.append(coords)

        return False


class MinimaxBoard(HexBoard):
    def __init__(self, board):
        self.size = board.size
        self.board = [row[:] for row in board.board]
        self.player_positions = {
            1: board.player_positions[1].copy(),
            2: board.player_positions[2].copy(),
        }

    def __hash__(self, value):
        return hash(
            (
                tuple(self.player_positions[1].copy()),
                tuple(self.player_positions[2].copy()),
            )
        )

    def __eq__(self, value):
        return self.player_positions == value.player_positions


class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)

    def play(self, board: HexBoard) -> tuple:
        raise NotImplementedError("¡Implementa este método!")


# implementation
class Monke(Player):
    def play(self, board):
        return random.choice(board.get_possible_moves())


class Usagi(Player):
    # rip, doesn't work after r-even
    def play(self, board):

        if self.player_id == 2:
            init = (0, 0)
            directions = [(1, 0), (0, 1), (1, -1)]
        else:
            init = (board.size - 1, board.size - 1)
            directions = [(0, -1), (-1, 0), (-1, 1)]

        while init in board.player_positions[self.player_id]:
            init += directions[0]

        for direction in directions:
            x, y = init[0] + direction[0], init[1] + direction[1]
            if board._unused_tile(x, y):
                return (x, y)

        return random.choice(board.get_possible_moves())


class AstarPlayer(Player):
    """Uses the A-star algo (duh). Explodes with self.size > 5"""

    def play(self, board):
        problem = HexProblem(board.clone(), self.player_id)
        solution_node = astar_search(problem)

        if solution_node in (failure, cutoff):
            raise Exception
        moves = path_actions(solution_node)
        move = random.choice(moves)

        return move


class MinimaxPlayer(Player):
    def __init__(self, player_id: int, depth=3, heuristic=manhattan):
        super().__init__(player_id)
        self.depth = depth
        self.heuristic = heuristic

    def play(self, board):
        mboard = MinimaxBoard(board)
        _, move = minimax(
            board,
            depth=self.depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximising=True,
            player_id=self.player_id,
            heuristic=manhattan,
        )
        return move

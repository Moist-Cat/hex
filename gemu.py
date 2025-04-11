from collections import deque
import math
import random
import concurrent.futures
from functools import partial

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
    full_distance_heuristic,
    distance_heuristic,
    average_distance_heuristic,
    adversarial_heuristic,
    opponent,
)

# abstract
class HexBoard:

    __slots__ = ("size", "board", "player_positions")

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
    def __init__(
        self,
        player_id: int,
        depth=2,
        # depth=3,
        # heuristic=lambda a, b: 0,
        # heuristic=average_distance_heuristic,
        # heuristic=distance_heuristic,
        heuristic=full_distance_heuristic,
        # heuristic=adversarial_heuristic(
        #    [full_distance_heuristic,]
        # )
    ):
        super().__init__(player_id)
        self.depth = depth
        self.heuristic = heuristic

    def play(self, board, time=0):
        mboard = MinimaxBoard(board)
        _, move = minimax(
            board,
            depth=self.depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximising=True,
            player_id=self.player_id,
            heuristic=self.heuristic,
        )
        if not move:
            # owarida..
            print("Giving up...")
            return mboard.get_possible_moves()[0]
        return move

    def __str__(self):
        return f"<Minimax (h={self.heuristic})>"


AlephNull = MinimaxPlayer


class MonteCarloPlayer(Player):
    """
    codename: Sun Wukong

    Infinite monkey theorem: a monkey hitting keys independently
       and at random on a typewriter keyboard for an infinite amount
       of time will almost surely type any given text,
       including the complete works of William Shakespeare.

    ... or, in our case, a monkey making random moves using all the compute power available will
        eventually win (spoiler: it doesn't).
    """

    def __init__(
        self,
        player_id,
        max_simulations=1000,
        min_simulations=100,
        win_check_interval=1,
        num_workers=1 * 8,
    ):
        super().__init__(player_id)
        self.max_simulations = max_simulations
        self.min_simulations = min_simulations
        self.win_check_interval = win_check_interval
        self.num_workers = num_workers

    def get_simulations_per_move(self, board):
        """Dynamically adjust simulations based on game progress"""
        num_played = sum(cell != 0 for row in board.board for cell in row)
        total_cells = board.size * board.size

        if num_played <= 5:
            return self.min_simulations
        elif num_played >= 10:
            return self.max_simulations
        # Linear interpolation between min and max
        # recall the values are between 5 and 10
        progress = (num_played - 5) / 5
        return int(
            self.min_simulations
            + (self.max_simulations - self.min_simulations) * progress
        )

    def simulate_playout(self, board, move):
        """Play at random and hope for the best"""
        sim_board = board.clone()
        sim_board.place_piece(*move, self.player_id)
        current_player = opponent(self.player_id)
        move_count = 1

        while True:
            valid_moves = sim_board.get_possible_moves()
            if not valid_moves:
                break

            # if len(sim_board.player_positions) < sim_board.size:
            #    top_moves = sorted(
            #        valid_moves,
            #          key=lambda m: self.get_move_priority(board, m),
            #          reverse=True
            #    )[:10]
            # else:
            top_moves = valid_moves

            chosen_move = random.choice(valid_moves)

            sim_board.place_piece(*chosen_move, current_player)
            move_count += 1

            # Periodic win check
            if move_count % self.win_check_interval == 0:
                if sim_board.check_connection(current_player):
                    return 1 if current_player == self.player_id else 0

            current_player = opponent(current_player)

        # In case we break (it won't happen)
        return 1 if sim_board.check_connection(self.player_id) else 0

    def evaluate_move(self, board, move, num_simulations):
        """Evaluate a move through parallel simulations"""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = [
                executor.submit(self.simulate_playout, board, move)
                for _ in range(num_simulations)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            return sum(results) / num_simulations  # win rate; if we didn't win we lost

    def get_move_priority(self, board, move):
        """Calculate priority score for move selection"""
        # Prefer center in early game, edges in mid-game
        size = board.size
        r, c = move
        center_dist = math.sqrt((r - size / 2) ** 2 + (c - size / 2) ** 2)

        if len(board.player_positions) < 5:
            return 1 / (1 + center_dist)  # Prefer center
        else:
            edge_dist = min(r, size - 1 - r, c, size - 1 - c)
            return edge_dist / size  # Prefer edges

    # overriden
    def get_move_priority(self, board, move):
        new_board = board.clone()
        x, y = move
        new_board.place_piece(x, y, self.player_id)

        return full_distance_heuristic(new_board, self.player_id)

    def play(self, board):
        valid_moves = board.get_possible_moves()
        if not valid_moves:
            return None

        # Special case: first move (prefer center)
        if all(cell == 0 for row in board.board for cell in row):
            return (board.size // 2, board.size // 2)

        simulations_per_move = self.get_simulations_per_move(board)
        move_stats = {}

        # First pass: quick evaluation of all moves using our heuristic
        # top_moves = sorted(
        #    valid_moves,
        #      key=lambda m: self.get_move_priority(board, m),
        #      reverse=True
        # )[:10]
        top_moves = valid_moves

        # Second pass: focus on promising moves
        for move in top_moves:
            move_stats[move] = {
                "win_rate": self.evaluate_move(board, move, simulations_per_move),
                "simulations": simulations_per_move,
            }

        for move in move_stats:
            if move_stats[move]["simulations"] > 0:
                move_stats[move]["score"] = move_stats[move]["win_rate"]
            else:
                move_stats[move]["score"] = 0

        # Select best move
        best_move = max(move_stats, key=lambda m: move_stats[m]["score"])
        return best_move


class SleepingDragon(Player):
    """
    Monte-Carlo + minmax hybrid.
    - Ultra defensive at the beginning
    - Utra aggresive near the end
    """

    def __init__(self, player_id):
        super().__init__(player_id)
        self.monkey = MonteCarloPlayer(
            player_id=player_id,
            max_simulations=100,
            min_simulations=10,
            win_check_interval=5,
            num_workers=100,
        )
        self.elephant = MinimaxPlayer(
            player_id=player_id,
            depth=2,
            heuristic=full_distance_heuristic,
        )

    def play(self, board, time=0):
        moves = len(board.player_positions[self.player_id])

        size = board.size

        if moves * 4 <= size:
            return self.elephant.play(board)
        else:
            # possibility to play m-c approaches 1 as the game advances
            # player = self.elephant if random.random() > min(1, (moves)/(size+1)) else self.monkey
            player = self.monkey
            return player.play(board)

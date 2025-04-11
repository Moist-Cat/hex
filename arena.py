import time
import statistics
import random
from gemu import HexBoard, AstarPlayer, Monke, Usagi, MinimaxPlayer as Candidate
from dsa import manhattan, full_distance_heuristic, adversarial_heuristic, average_distance_heuristic


class Arena:
    def __init__(self, candidate, levels, board_size=4, games_per_level=10, **candidate_kwargs):
        self.candidate = candidate
        candidate._is_candidate = True
        self.levels = levels
        self.board_size = board_size
        self.games_per_level = games_per_level
        self.report = []
        self.candidate_kwargs = candidate_kwargs

    def _play_game(self, candidate_class, opponent_class):
        board = HexBoard(size=self.board_size)
        if random.random() > 0.5:
            candidate = candidate_class(player_id=1, **self.candidate_kwargs)
            opponent = opponent_class(player_id=2)
        else:
            candidate = candidate_class(player_id=2, **self.candidate_kwargs)
            opponent = opponent_class(player_id=1)
        opponent._is_candidate = False

        move_times = []
        winner = None
        current_player = candidate
        move_count = 0

        while True:
            # time only candidate's moves
            if (
                current_player._is_candidate
            ):
                start_time = time.perf_counter()
                move = current_player.play(board)
                elapsed = time.perf_counter() - start_time
                move_times.append(elapsed)
            else:
                move = current_player.play(board)

            row, col = move
            if not board.place_piece(row, col, current_player.player_id):
                raise ValueError(
                    f"Invalid move by {type(current_player).__name__} at {move}"
                )

            # Check win condition
            if board.check_connection(current_player.player_id):
                winner = current_player
                break

            # Switch players
            if current_player is candidate:
                current_player = opponent
            else:
                current_player = candidate
            move_count += 1

        return {
            "winner": winner,
            "first": candidate.player_id == 1,
            "move_times": move_times,
            "total_moves": move_count + 1,
        }

    def run(self):
        """Run the complete tournament"""
        for level_num, Opponent in enumerate(self.levels):
            level_name = Opponent.__name__
            print(f"\n=== Testing against {level_name} ===")

            wins = 0
            all_move_times = []
            game_metrics = []

            for game_num in range(1, self.games_per_level + 1):
                result = self._play_game(self.candidate, Opponent)

                if result["winner"]._is_candidate:
                    wins += 1

                game_metrics.append(
                    {"moves": result["total_moves"], "time": sum(result["move_times"]), "first": result["first"]}
                )
                all_move_times.extend(result["move_times"])

                print(
                    f"Game {game_num}: {'Won' if result['winner']._is_candidate else 'Lost'} "
                    f"in {result['total_moves']} moves "
                    f"(Decision time: {sum(result['move_times']):.2f}s)"
                )

            # Calculate statistics
            win_rate = wins / self.games_per_level
            total_time = sum(gm["time"] for gm in game_metrics)
            avg_move_time = statistics.mean(all_move_times) if all_move_times else 0
            time_per_game = statistics.mean(gm["time"] for gm in game_metrics)
            moves_per_game = statistics.mean(gm["moves"] for gm in game_metrics)
            first_rate = sum(gm["first"] for gm in game_metrics)

            self.report.append(
                {
                    "Level": level_name,
                    "Win Rate": f"{win_rate:.0%}",
                    "Total Time": f"{total_time:.2f}s",
                    "Avg Move Time": f"{avg_move_time:.3f}s ",
                    "Moves/Game": f"{moves_per_game:.1f}",
                    "Time/Game": f"{time_per_game:.2f}s",
                    "Times first": f"{(first_rate/self.games_per_level)*100}%",
                    "Passed": win_rate >= 0.5,
                }
            )

            if not self.report[-1]["Passed"]:
                print("\n🚫 Failed to pass this level")
                break
            else:
                print("\n✅ Passed this level")

        return self.report

    def print_report(self):
        """Display formatted results"""
        print("\n=== Final Tournament Report ===")
        for entry in self.report:
            print(f"\nLevel: {entry['Level']}")
            print(f"  Win Rate:       {entry['Win Rate']}")
            print(f"  Total Time:     {entry['Total Time']}")
            print(f"  Avg Move Time:  {entry['Avg Move Time']}")
            print(f"  Avg Moves/Game: {entry['Moves/Game']}")
            print(f"  Avg Time/Game:  {entry['Time/Game']}")
            print(f"  Times first:  {entry['Times first']}")
            print(f"  Passed:         {'✅' if entry['Passed'] else '❌'}")


# Usage example
arena = Arena(
    candidate=Candidate,
    #levels=[Monke, Usagi, AstarPlayer, Candidate],
    levels=[Monke, Usagi, Candidate],
    #levels=[Monke,],
    #levels=[Candidate,],
    board_size=11,
    games_per_level=10,
    #heuristic=adversarial_heuristic([full_distance_heuristic,])
    heuristic=full_distance_heuristic,
    #heuristic=average_distance_heuristic,
)

report = arena.run()
arena.print_report()

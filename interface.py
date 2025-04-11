import time
from gemu import HexBoard, MinimaxPlayer as Candidate, Monke as Other, MonteCarloPlayer, SleepingDragon
from dsa import adversarial_heuristic, full_distance_heuristic, average_distance_heuristic

class HexGameCLI:
    def __init__(self):
        self.board_size = 11
        self.human_player = 1
        self.ai_player = 2
        #self.ai = SleepingDragon
        self.ai = Candidate
        self.ai_2 = Other
        #self.ai_2 = MinimaxPlayer#Other
        #self.ai_2 = Candidate
        self.vs_ai = True or False
        self.ai_metrics = {
            'move_times': [],
            'avg_move_time': 0,
            'wins': 0,
            'games_played': 0
        }
        
    def clear_screen(self):
        print("\033[H\033[J")  # ANSI escape codes for clearing screen

    def print_board(self, board):
        """Print the hex board with ASCII art"""
        # we don't deepcopy
        board._sync()
        print("\n   " + " ".join(f"{i:2}" for i in range(board.size)))
        for r in range(board.size):
            # Offset odd rows for hex layout
            row_prefix = "  "*r
            print(f"{r:2}{row_prefix}", end="")
            
            for c in range(board.size):
                cell = board.board[r][c]
                if cell == 0:
                    print(" . ", end="")
                elif cell == 1:
                    print(" X ", end="")  # Human
                else:
                    print(" O ", end="")  # AI
            print()

    def get_human_move(self):
        while True:
            try:
                move = input("Your move (row col): ").strip().split()
                if len(move) != 2:
                    raise ValueError
                row, col = map(int, move)
                if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                    raise ValueError
                return (row, col)
            except ValueError:
                print(f"Invalid input! Enter two numbers 0-{self.board_size-1} separated by space.")

    def show_stats(self):
        """Display AI performance metrics"""
        print("\n=== AI Statistics ===")
        print(f"Games Played: {self.ai_metrics['games_played']}")
        print(f"AI Wins: {self.ai_metrics['wins']}")
        if self.ai_metrics['move_times']:
            avg = sum(self.ai_metrics['move_times'])/len(self.ai_metrics['move_times'])
            print(f"Avg Move Time: {avg:.2f}s")
        print("===================")

    def play_game(self):
        board = HexBoard(size=self.board_size)
        ai = self.ai(player_id=self.ai_player)
        if self.vs_ai:
            #ai_2 = self.ai_2(player_id=self.human_player, heuristic=adversarial_heuristic([full_distance_heuristic]))
            #ai_2 = self.ai_2(player_id=self.human_player, heuristic=full_distance_heuristic)
            ai_2 = self.ai_2(player_id=self.human_player)
            #ai_2 = self.ai_2(player_id=self.human_player, heuristic=average_distance_heuristic)
            #ai_2 = self.ai_2(player_id=self.human_player, heuristic=full_distance_heuristic, depth=3)
            #ai_2 = self.ai_2(player_id=self.human_player, heuristic=lambda a,b: 0)
        current_player = 1  # Human starts first
        
        while True:
            self.clear_screen()
            print(f"Hex Game (Size: {self.board_size}) - You are {'X (Player 1)' if self.human_player == 1 else 'O (Player 2)'}")
            self.print_board(board)
            self.show_stats()
            
            if current_player == self.human_player:
                # Human turn
                if not self.vs_ai:
                    row, col = self.get_human_move()
                else:
                    row, col = ai_2.play(board)
                if not board.place_piece(row, col, self.human_player):
                    print("Invalid move! Try again.")
                    time.sleep(1)
                    continue
            else:
                # AI turn
                print("\nAI thinking...")
                start_time = time.time()
                row, col = ai.play(board)
                move_time = time.time() - start_time
                self.ai_metrics['move_times'].append(move_time)
                board.place_piece(row, col, self.ai_player)
                print(f"AI played: {row} {col} (took {move_time:.2f}s)")
                time.sleep(0.5)  # Pause to see AI move
            
            # Check win condition
            if board.check_connection(current_player):
                self.clear_screen()
                self.print_board(board)
                winner = "You" if current_player == self.human_player else "AI"
                print(f"\n{winner} won!")
                if winner == "AI":
                    self.ai_metrics['wins'] += 1
                self.ai_metrics['games_played'] += 1
                #input("Press any key to continue...")
                return
            
            current_player = 3 - current_player  # Switch player (1 ↔ 2)

    def settings_menu(self):
        self.clear_screen()
        print("=== Hex Game Settings ===")
        print(f"1. Board Size (Current: {self.board_size})")
        print(f"2. Player Side (Current: {'Player 1 (X)' if self.human_player == 1 else 'Player 2 (O)'})")
        print("3. Start Game")
        print(f"4. Toggle VS AI (Current: {self.vs_ai})")
        print("5. Exit")
        
        choice = input("Select: ").strip()
        if choice == "1":
            self.board_size = int(input("Enter board size: "))
        elif choice == "2":
            self.human_player = 2 if self.human_player == 1 else 1
            self.ai_player = 3 - self.human_player
        elif choice == "3":
            self.play_game()
        elif choice == "4":
            self.vs_ai ^= self.vs_ai
        elif choice == "5":
            exit()
        else:
            print("Invalid choice!")
            time.sleep(1)

    def main_menu(self):
        while True:
            self.clear_screen()
            print("=== Hex Game ===")
            print("1. Play vs AI")
            print("2. Settings")
            print("3. Exit")
            
            choice = input("Select: ").strip()
            if choice == "1":
                self.play_game()
                input("\nPress Enter to continue...")
            elif choice == "2":
                self.settings_menu()
            elif choice == "3":
                exit()
            else:
                print("Invalid choice!")
                time.sleep(1)

if __name__ == "__main__":
    game = HexGameCLI()
    game.main_menu()

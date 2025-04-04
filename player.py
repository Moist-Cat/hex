from gemu import AstarPlayer, HexBoard, Monke, Usagi

board = HexBoard(size=4)
ai_player = AstarPlayer(player_id=1)
move = ai_player.play(board)
print(move)
ai_player = Monke(player_id=2)
move = ai_player.play(board)
print(move)
ai_player = Usagi(player_id=1)
move = ai_player.play(board)
print(move)
board._sync()
print(board.board, board.player_positions)

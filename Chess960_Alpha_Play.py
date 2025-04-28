import chess
import chess.engine
import numpy as np
import torch
from AlphaZeroChess960 import Chess960Game, ResNet, MCTS

# Play loops with explicit move construction

def human_vs_model(game, model, mcts, board):
    """Human (White) vs Model (Black)"""
    while True:
        # Human move
        print(board)
        move_uci = input("Your move (UCI): ").strip()
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            print("  ✗ Invalid UCI. Try again.")
            continue
        if move not in board.legal_moves:
            print("  ✗ Illegal move. Try again.")
            continue
        board.push(move)
        val, done = game.get_value_and_terminated(board)
        if done:
            print(board)
            result = "White wins" if val>0 else ("Draw" if val==0 else "Black wins")
            print("→ Game over:", result)
            break

        # Model move
        print("\nModel thinking (Black)...\n")
        probs = mcts.search(board)
        action = int(np.argmax(probs))
        from_sq, to_sq = divmod(action, 64)
        mv = chess.Move(from_sq, to_sq)
        if mv not in board.legal_moves:
            mv = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        print("Model plays:", mv.uci(), "\n")
        board.push(mv)
        val, done = game.get_value_and_terminated(board)
        if done:
            print(board)
            result = "White wins" if val>0 else ("Draw" if val==0 else "Black wins")
            print("→ Game over:", result)
            break


def model_vs_human(game, model, mcts, board):
    """Model (White) vs Human (Black)"""
    while True:
        # Model move
        print("\nModel thinking (White)...\n")
        probs = mcts.search(board)
        action = int(np.argmax(probs))
        from_sq, to_sq = divmod(action, 64)
        mv = chess.Move(from_sq, to_sq)
        if mv not in board.legal_moves:
            mv = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        print("Model plays:", mv.uci(), "\n")
        board.push(mv)
        val, done = game.get_value_and_terminated(board)
        if done:
            print(board)
            result = "White wins" if val>0 else ("Draw" if val==0 else "Black wins")
            print("→ Game over:", result)
            break

        # Human move
        print(board)
        move_uci = input("Your move (UCI): ").strip()
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            print("  ✗ Invalid UCI. Try again.")
            continue
        if move not in board.legal_moves:
            print("  ✗ Illegal move. Try again.")
            continue
        board.push(move)
        val, done = game.get_value_and_terminated(board)
        if done:
            print(board)
            result = "White wins" if val>0 else ("Draw" if val==0 else "Black wins")
            print("→ Game over:", result)
            break


def model_vs_stockfish(game, model, mcts, engine, board, time_limit=0.1):
    """Model (White) vs Stockfish (Black)"""
    while True:
        # Model move
        print("\nModel thinking (White)...\n")
        probs = mcts.search(board)
        action = int(np.argmax(probs))
        from_sq, to_sq = divmod(action, 64)
        mv = chess.Move(from_sq, to_sq)
        if mv not in board.legal_moves:
            mv = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        print("Model plays:", mv.uci(), "\n")
        board.push(mv)
        if board.is_game_over():
            print(board)
            print("→ Game over:", board.result())
            break

        # Stockfish move
        print(board)
        print("\nStockfish thinking (Black)...\n")
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        print("Stockfish plays:", result.move.uci(), "\n")
        board.push(result.move)
        if board.is_game_over():
            print(board)
            print("→ Game over:", board.result())
            break


def stockfish_vs_model(game, model, mcts, engine, board, time_limit=0.1):
    """Stockfish (White) vs Model (Black)"""
    while True:
        # Stockfish move
        print("\nStockfish thinking (White)...\n")
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        print("Stockfish plays:", result.move.uci(), "\n")
        board.push(result.move)
        if board.is_game_over():
            print(board)
            print("→ Game over:", board.result())
            break

        # Model move
        print(board)
        print("\nModel thinking (Black)...\n")
        probs = mcts.search(board)
        action = int(np.argmax(probs))
        from_sq, to_sq = divmod(action, 64)
        mv = chess.Move(from_sq, to_sq)
        if mv not in board.legal_moves:
            mv = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        print("Model plays:", mv.uci(), "\n")
        board.push(mv)
        val, done = game.get_value_and_terminated(board)
        if done:
            print(board)
            result = "White wins" if val>0 else ("Draw" if val==0 else "Black wins")
            print("→ Game over:", result)
            break


def main():
    # Init game & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game   = Chess960Game()
    model = ResNet(game).to(device)
    checkpoint = torch.load('./models/az_model_toy.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # MCTS
    args = {'C':1.4, 'num_searches':50, 'epsilon':0.00, 'dirichlet_alpha':0.03}
    mcts = MCTS(game, model, args, device)

    # Stockfish
    sf_path = "/usr/local/bin/stockfish"
    engine  = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": 1500})

    # Menu: mode + colors
    print("Choose mode:")
    print("  1: Human vs Model")
    print("  2: Model vs Stockfish")
    mode = input("Enter 1 or 2: ").strip()

    board = game.get_initial_state()
    print("Starting Chess960 position:")
    print(board)
    print("FEN:", board.fen(), "\n")

    if mode == '1':
        hc = input("Should human play White or Black? (W/B): ").strip().upper()
        if hc == 'W':
            human_vs_model(game, model, mcts, board)
        else:
            model_vs_human(game, model, mcts, board)
    elif mode == '2':
        mc = input("Should model play White or Black? (W/B): ").strip().upper()
        if mc == 'W':
            model_vs_stockfish(game, model, mcts, engine, board)
        else:
            stockfish_vs_model(game, model, mcts, engine, board)
    else:
        print("Invalid choice. Exiting.")

    engine.quit()

if __name__ == '__main__':
    main()

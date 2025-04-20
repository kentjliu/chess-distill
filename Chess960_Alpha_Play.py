import chess
import numpy as np
import torch
from AlphaZeroChess960 import Chess960Game, ResNet, MCTS
#Very simple method for actually playing the game. 
'''Make sure that the model you load in is the same as the model 
in this class or this will not work. Changes in the Alphazero
exported model need to be reflected here!'''

def main():
    # 1) Init game & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game   = Chess960Game()

    model = ResNet(game).to(device)
    checkpoint = torch.load('/content/az_model_0.pt', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2) MCTS wrapper with fewer rollouts for interactive speed
    args = {
        'C':               1.4,
        'num_searches':   50,    # you can lower for faster moves
        'epsilon':       0.00,   # no Dirichlet noise in play
        'dirichlet_alpha':0.03,
    }
    mcts = MCTS(game, model, args, device)

    # 3) Start a fresh Chess960 position
    board = game.get_initial_state()
    print("Starting Chess960 position:")
    print(board)  
    print("FEN:", board.fen(), "\n")

    # 4) Game loop
    while True:
        # --- Your move (White) ---
        print(board)  
        move_uci = input("Your move (UCI, e.g. e2e4): ").strip()
        try:
            move = chess.Move.from_uci(move_uci)
        except:
            print("  ✗ Invalid UCI format. Try again.")
            continue
        if move not in board.legal_moves:
            print("  ✗ Illegal move. Try again.")
            continue
        board.push(move)

        # check for game end
        val, done = game.get_value_and_terminated(board)
        if done:
            print(board)
            result = "White wins" if val>0 else ("Draw" if val==0 else "Black wins")
            print("→ Game over:", result)
            break

        # --- Model move (Black) ---
        print("\nModel thinking...\n")
        probs = mcts.search(board)
        action = int(np.argmax(probs))
        from_sq, to_sq = action // 64, action % 64
        mv = chess.Move(from_sq, to_sq)
        if mv not in board.legal_moves:
            # maybe promotion
            mv = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        print("Model plays:", mv.uci(), "\n")
        board = game.get_next_state(board, action)

        val, done = game.get_value_and_terminated(board)
        if done:
            print(board)
            result = "White wins" if val>0 else ("Draw" if val==0 else "Black wins")
            print("→ Game over:", result)
            break

if __name__ == '__main__':
    main()

import os
import chess
import chess.pgn

# Directory containing extracted PGN files
pgn_dir = "lichess_chess960"
training_data_file = "chess960_training_data.txt"

def parse_pgn_and_generate_pairs(pgn_path):
    """Parse PGN files, extract moves, and generate state-action pairs."""
    state_action_pairs = []
    
    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break  # End of file

            # Get Chess960 starting position from PGN
            if "FEN" in game.headers:
                start_fen = game.headers["FEN"]
                board = chess.Board(fen=start_fen)  # Initialize board with Chess 960 position
            else:
                board = chess.Board()  # Default to standard chess if no FEN is provided
            
            for move in game.mainline_moves():
                if board.is_game_over():
                    break  # Stop if game is over

                state = board.fen()  # Get board state as FEN
                state_action_pairs.append(f"{state} {move.uci()}")  # Store state and action

                board.push(move)  # Apply move
                
    return state_action_pairs

# Process all PGN files and store training data
all_pairs = []
for filename in os.listdir(pgn_dir):
    if filename.endswith(".pgn"):
        pgn_path = os.path.join(pgn_dir, filename)
        all_pairs.extend(parse_pgn_and_generate_pairs(pgn_path))
        print(f"Processed: {filename}")
        break

# Save training data
with open(training_data_file, "w", encoding="utf-8") as f:
    for pair in all_pairs:
        f.write(pair + "\n")

print(f"Saved {len(all_pairs)} state-action pairs to {training_data_file}")

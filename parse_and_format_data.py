import os
import chess.pgn

# Directory containing extracted PGN files
pgn_dir = "lichess_chess960"
parsed_games_file = "parsed_chess960_games.txt"

def parse_pgn_file(pgn_path):
    """Parse a PGN file and extract moves."""
    games = []
    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break  # End of file
            moves = [move.uci() for move in game.mainline_moves()]
            games.append(" ".join(moves))  # Store as a UCI move sequence
    return games

# Process all PGN files and store moves
all_games = []
for filename in os.listdir(pgn_dir):
    if filename.endswith(".pgn"):
        pgn_path = os.path.join(pgn_dir, filename)
        all_games.extend(parse_pgn_file(pgn_path))
        print(f"Parsed: {filename}")

# Save parsed games
with open(parsed_games_file, "w", encoding="utf-8") as f:
    for game in all_games:
        f.write(game + "\n")

print(f"Saved {len(all_games)} games to {parsed_games_file}")

import chess
import chess.pgn

parsed_games_file = "parsed_chess960_games.txt"
training_data_file = "chess960_training_data.txt"

def generate_state_action_pairs(moves_list):
    """Convert a sequence of UCI moves into state-action pairs."""
    state_action_pairs = []
    board = chess.Board()  # Starts as an empty Chess960 board

    for move in moves_list:
        if board.is_game_over():
            break  # Stop if game is over

        state = board.fen()  # Get board state as FEN
        state_action_pairs.append(f"{state} {move}")  # Save state and action

        board.push_uci(move)  # Apply move

    return state_action_pairs

# Read parsed games and generate state-action pairs
all_pairs = []
with open(parsed_games_file, "r", encoding="utf-8") as f:
    for line in f:
        moves = line.strip().split()
        all_pairs.extend(generate_state_action_pairs(moves))

# Save training data
with open(training_data_file, "w", encoding="utf-8") as f:
    for pair in all_pairs:
        f.write(pair + "\n")

print(f"Saved {len(all_pairs)} state-action pairs to {training_data_file}")


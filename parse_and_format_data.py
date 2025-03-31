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

#!/usr/bin/env python3
"""
play_eval.py  – benchmark two AlphaZero-style models against Stockfish.

-------------------
• Tracks *pieces won* (number of opponent pieces the model captures in a game)
  and reports the average per game alongside depth and W-D-L.

Example
-------
python play_eval.py --games 100 \
    --model1 ./models/az_model_25.pth \
    --model2 ./models/az_model_15.pth \
    --stockfish /opt/homebrew/bin/stockfish
"""

import argparse, os, statistics, time
import chess, chess.engine
import numpy as np
import torch
from tqdm import tqdm

from AlphaZeroChess960 import Chess960Game, ResNet, MCTS    # type: ignore


def load_model(game, ckpt_path, device):
    model = ResNet(game).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    own_state = model.state_dict()
    filtered = {}
    for name, param in state_dict.items():
        if name in own_state:
            if param.size() == own_state[name].size():
                filtered[name] = param
            else:
                print(f"[load_model] SKIP {name}: checkpoint shape {tuple(param.size())} ≠ model shape {tuple(own_state[name].size())}")
        else:
            print(f"[load_model] IGNORE {name}: not in model")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[load_model] missing keys (left at random init): {missing}")
    if unexpected:
        print(f"[load_model] unexpected keys: {unexpected}")

    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    size_mb  = os.path.getsize(ckpt_path) / (1024**2)
    return model, n_params, size_mb


def pick_model_move(board, mcts):
    probs  = mcts.search(board)
    action = int(np.argmax(probs))
    frm, to = divmod(action, 64)
    mv = chess.Move(frm, to)
    if mv not in board.legal_moves:
        mv = chess.Move(frm, to, promotion=chess.QUEEN)
    return mv


def play_one_game(game, mcts, engine, model_plays_white,
                  max_moves=300, time_limit=0.1):
    """
    Plays a single game.

    Returns
    -------
    outcome : str   -- PGN result string ("1-0", "0-1", "1/2-1/2", "*")
    depth   : int   -- full-move count
    captures: int   -- pieces won by the model
    """
    board = game.get_initial_state()
    depth = 0
    captures = 0

    while not board.is_game_over(claim_draw=True) and depth < max_moves:
        model_to_move = (board.turn and model_plays_white) or \
                        (not board.turn and not model_plays_white)

        if model_to_move:
            mv = pick_model_move(board, mcts)
            if board.is_capture(mv):
                captures += 1
        else:
            mv = engine.play(board, chess.engine.Limit(time=time_limit)).move

        board.push(mv)
        if board.turn:
            depth += 1

    return board.result(claim_draw=True), depth, captures


def run_matches(model_name, mcts, n_games, engine, game):
    results = {"win": 0, "loss": 0, "draw": 0}
    depths, caps = [], []

    for g in tqdm(range(n_games)):
        model_white = (g % 2 == 0)
        outcome, depth, captures = play_one_game(
            game, mcts, engine, model_white)

        depths.append(depth)
        caps.append(captures)

        if outcome == "1-0":
            if model_white: results["win"]  += 1
            else:           results["loss"] += 1
        elif outcome == "0-1":
            if model_white: results["loss"] += 1
            else:           results["win"]  += 1
        else:
            results["draw"] += 1

    return (results,
            statistics.mean(depths),
            statistics.mean(caps))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games",      type=int,   default=100)
    p.add_argument("--model1",     type=str,   default="./models/az_model_25.pth")
    p.add_argument("--model2",     type=str,   default="./models/az_model_15.pth")
    p.add_argument("--stockfish",  type=str,   default="/opt/homebrew/bin/stockfish")
    p.add_argument("--searches",   type=int,   default=50)
    p.add_argument("--sf-elo",     type=int,   default=1500)
    p.add_argument("--time",       type=float, default=0.1,
                   help="Stockfish time limit (seconds) per move")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game   = Chess960Game()

    # Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.sf_elo})

    # Model 1
    model1, p1, sz1 = load_model(game, args.model1, device)
    mcts1 = MCTS(game, model1,
                 {"C": 1.4,
                  "num_searches": args.searches,
                  "epsilon": 0.0,
                  "dirichlet_alpha": 0.03},
                 device)

    # Model 2
    model2, p2, sz2 = load_model(game, args.model2, device)
    mcts2 = MCTS(game, model2,
                 {"C": 1.4,
                  "num_searches": args.searches,
                  "epsilon": 0.0,
                  "dirichlet_alpha": 0.03},
                 device)

    print("────────────────────────────────────────────────────────────")
    print(f"Model-1: {args.model1} | params: {p1:,} | file: {sz1:.1f} MB")
    print(f"Model-2: {args.model2} | params: {p2:,} | file: {sz2:.1f} MB")
    print(f"Games per model: {args.games}  |  Stockfish Elo {args.sf_elo}")
    print("────────────────────────────────────────────────────────────")

    t0 = time.time()
    res1, depth1, cap1 = run_matches("Model-1", mcts1, args.games, engine, game)
    res2, depth2, cap2 = run_matches("Model-2", mcts2, args.games, engine, game)
    elapsed = time.time() - t0
    engine.quit()

    fmt = lambda r: f"{r['win']}-{r['draw']}-{r['loss']}"   # W-D-L
    print("\n══════════════════════  RESULTS  ══════════════════════")
    print(f"Model-1  W-D-L: {fmt(res1)}  | avg depth: {depth1:.1f}"
          f"  | avg pieces won: {cap1:.1f}")
    print(f"Model-2  W-D-L: {fmt(res2)}  | avg depth: {depth2:.1f}"
          f"  | avg pieces won: {cap2:.1f}")
    print(f"Total runtime: {elapsed/60:.1f} min")
    print("════════════════════════════════════════════════════════")

if __name__ == "__main__":
    main()

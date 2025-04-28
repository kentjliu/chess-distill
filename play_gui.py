import os, sys, pygame, chess, chess.engine, numpy as np, torch
from AlphaZeroChess960 import Chess960Game, ResNet, MCTS

# ──── GUI CONSTANTS ─────────────────────────────────────────────────────────────
TILE = 80                                      # square size in pixels
LIGHT = (240, 217, 181)                        # board colours
DARK  = (181, 136,  99)
HLAST = (246, 246, 105)                        # last-move highlight

# ──── UTILITIES ─────────────────────────────────────────────────────────────────
def load_piece_images():
    imgs = {}
    for c in "bw":
        for p in "prnbqk":
            fn = os.path.join("assets", f"{c}{p}.png")
            if not os.path.exists(fn):
                sys.exit(f"Missing piece image: {fn}")
            img = pygame.image.load(fn)
            imgs[f"{c}{p}"] = pygame.transform.smoothscale(img, (TILE, TILE))
    return imgs

def square_from_mouse():
    mx, my = pygame.mouse.get_pos()
    if 0 <= mx < 8*TILE and 0 <= my < 8*TILE:
        file, rank = mx // TILE, 7 - (my // TILE)
        return chess.square(file, rank)
    return None

def draw_board(screen, board, imgs, last=None, drag_img=None, drag_pos=None):
    # squares
    for sq in chess.SQUARES:
        col = LIGHT if (sq + chess.square_rank(sq)) & 1 == 0 else DARK
        x, y = chess.square_file(sq)*TILE, (7-chess.square_rank(sq))*TILE
        pygame.draw.rect(screen, col, (x, y, TILE, TILE))
    # last move highlight
    if last:
        for sq in [last.from_square, last.to_square]:
            x, y = chess.square_file(sq)*TILE, (7-chess.square_rank(sq))*TILE
            pygame.draw.rect(screen, HLAST, (x, y, TILE, TILE))
    # pieces
    for sq, piece in board.piece_map().items():
        if drag_img and sq == last and drag_pos:   # skip the piece we are dragging
            continue
        key = f'{"b" if piece.color else "w"}{piece.symbol().lower()}'
        x, y = chess.square_file(sq)*TILE, (7-chess.square_rank(sq))*TILE
        screen.blit(imgs[key], (x, y))
    # dragged piece on top
    if drag_img and drag_pos:
        screen.blit(drag_img, drag_pos)

# ──── BACK-END HELPER (engine or model move) ───────────────────────────────────
def pick_model_move(board, mcts):
    probs  = mcts.search(board)
    action = int(np.argmax(probs))
    frm, to = divmod(action, 64)
    mv = chess.Move(frm, to)
    if mv not in board.legal_moves:
        mv = chess.Move(frm, to, promotion=chess.QUEEN)
    return mv

# ──── MAIN GUI LOOP ────────────────────────────────────────────────────────────
def play_with_gui(mode, human_is_white, model_is_white,
                  game, mcts, engine=None, time_limit=0.1):
    # pygame setup
    pygame.init()
    screen  = pygame.display.set_mode((8*TILE, 8*TILE))
    pygame.display.set_caption("Chess-960")
    clock   = pygame.time.Clock()
    imgs    = load_piece_images()

    board     = game.get_initial_state()
    dragging  = False
    drag_from = None
    drag_img  = None
    last_move = None
    result    = None

    while True:
        clock.tick(60)
        human_turn = (board.turn and human_is_white) or (not board.turn and not human_is_white)

        # ── HANDLE EVENTS ────────────────────────────────────────────────────
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit(); return
            if human_turn:
                if evt.type == pygame.MOUSEBUTTONDOWN and not dragging:
                    sq = square_from_mouse()
                    if sq is not None and board.piece_at(sq) and board.turn == board.piece_at(sq).color:
                        dragging  = True
                        drag_from = sq
                        pc        = board.piece_at(sq)
                        drag_img  = imgs[f'{"b" if pc.color else "w"}{pc.symbol().lower()}']
                elif evt.type == pygame.MOUSEBUTTONUP and dragging:
                    dst = square_from_mouse()
                    dragging = False
                    if dst is not None:
                        mv = chess.Move(drag_from, dst)
                        if mv not in board.legal_moves:
                            mv = chess.Move(drag_from, dst, promotion=chess.QUEEN)
                        if mv in board.legal_moves:
                            board.push(mv); last_move = mv
                            v, done = game.get_value_and_terminated(board)
                            if done:
                                result = "White wins" if v>0 else ("Draw" if v==0 else "Black wins")
                    drag_img = None

        # ── ENGINE ACTIONS (if it is their turn and no GUI interaction) ─────
        if not board.is_game_over():
            if (board.turn and not human_is_white) or (not board.turn and human_is_white):
                # model plays
                mv = pick_model_move(board, mcts)
                board.push(mv); last_move = mv
            elif mode == "engine_vs_model" and ((board.turn and not model_is_white) or
                                                (not board.turn and model_is_white)):
                # stockfish plays
                mv = engine.play(board, chess.engine.Limit(time=time_limit)).move
                board.push(mv); last_move = mv

        # ── TERMINATION CHECK ───────────────────────────────────────────────
        if board.is_game_over() and result is None:
            result = board.result() if mode == "engine_vs_model" else (
                     "White wins" if board.result() == "1-0" else ("Draw" if board.result() == "1/2-1/2" else "Black wins"))

        # ── DRAW EVERYTHING ────────────────────────────────────────────────
        draw_pos = pygame.mouse.get_pos() if dragging else None
        draw_board(screen, board, imgs, last_move,
                   drag_img if dragging else None, draw_pos)
        pygame.display.flip()

        if result:
            print("→ Game over:", result)
            pygame.time.wait(2000)
            pygame.quit()
            return

# ──── ENTRY POINT (keeps your menu & model setup) ─────────────────────────────
def main():
    # back-end init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game   = Chess960Game()
    model = ResNet(game).to(device)
    checkpoint = torch.load('./models/az_model_toy.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    mcts   = MCTS(game, model, {'C':1.4,'num_searches':50,'epsilon':0,'dirichlet_alpha':0.03}, device)

    # menu
    print("Choose mode:")
    print("  1: Human vs Model")
    print("  2: Model vs Stockfish")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        hc = input("Should human play White or Black? (W/B): ").strip().upper()
        play_with_gui("human_vs_model",
                      human_is_white = (hc=="W"),
                      model_is_white = (hc!="W"),
                      game=game, mcts=mcts)
    elif mode == "2":
        mc = input("Should model play White or Black? (W/B): ").strip().upper()
        stockfish = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
        stockfish.configure({"UCI_LimitStrength":True, "UCI_Elo":1500})
        play_with_gui("engine_vs_model",
                      human_is_white = False,
                      model_is_white = (mc=="W"),
                      game=game, mcts=mcts,
                      engine=stockfish)
        stockfish.quit()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
import chess
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Chess960 game definition ---
class Chess960Game:
    def __init__(self):
        # 8x8 board, action = from_square(0-63) * 64 + to_square(0-63)
        self.board_size = 8
        self.action_size = self.board_size * self.board_size
        self.action_size = self.action_size * self.action_size  # 64*64

    def get_initial_state(self):
        # start from a random Chess960 position
        idx = random.randint(0, 959)
        return chess.Board.from_chess960_pos(idx)

    def get_valid_moves(self, board: chess.Board) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=np.uint8)
        for move in board.legal_moves:
            idx = move.from_square * 64 + move.to_square
            mask[idx] = 1
        return mask

    def get_next_state(self, board: chess.Board, action: int) -> chess.Board:
        from_sq = action // 64
        to_sq   = action % 64
        new_board = board.copy()
        piece = new_board.piece_at(from_sq)

        # Try the normal move first
        plain = chess.Move(from_sq, to_sq)
        if plain in new_board.legal_moves:
            new_board.push(plain)
            return new_board

        # Only consider promotion if it’s a pawn to the back rank
        is_pawn      = piece and piece.piece_type == chess.PAWN
        rank_to_back = chess.square_rank(to_sq) in (0, 7)
        if is_pawn and rank_to_back:
            promo = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
            if promo in new_board.legal_moves:
                new_board.push(promo)
                return new_board

        # Otherwise it really is illegal
        raise ValueError(f"Illegal action {plain} (or promotion) on position:\n{board.fen()}")

    def get_value_and_terminated(self, board: chess.Board) -> (float, bool):
        if board.is_checkmate():
            # winner is opponent of current turn
            winner = board.turn ^ True
            value = 1.0 if winner == chess.WHITE else -1.0
            return value, True
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0, True
        return 0.0, False

    def get_opponent_value(self, value: float) -> float:
        return -value

    def change_perspective(self, board: chess.Board) -> chess.Board:
        # rotate board so that current player is always white in encoding
        # swap white/black pieces
        flipped = board.mirror()
        flipped.turn = True
        return flipped

    def get_encoded_state(self, board: chess.Board) -> np.ndarray:
        # 12x8x8 planes: white P,N,B,R,Q,K and black P,N,B,R,Q,K
        encoded = np.zeros((18, 8, 8), dtype=np.float32)
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
                rank = chess.square_rank(sq)
                file = chess.square_file(sq)
                row = 7 - rank
                col = file
                encoded[idx, row, col] = 1.0
        return encoded

# --- Neural network definition ---
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class ResNet(nn.Module):
    def __init__(self, game: Chess960Game, num_input_planes=18, num_resBlocks=4, num_hidden=128):
        super().__init__()
        # updated input layer to accept 18 channels
        self.startBlock = nn.Sequential(
            nn.Conv2d(num_input_planes, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backBone = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_resBlocks)])
        # policy head outputs 4096 logits
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.board_size * game.board_size, game.action_size)
        )
        # value head
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * game.board_size * game.board_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        return self.policyHead(x), self.valueHead(x)

# --- MCTS with neural guidance ---
class Node:
    def __init__(self, game, state, parent=None, prior=0.0):
        self.game = game
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, policy):
        for a, p in enumerate(policy):
            if p > 0:
                self.children[a] = Node(self.game,
                                        self.game.get_next_state(self.state, a),
                                        parent=self,
                                        prior=p)

    def select(self, C):
        best_score = -float('inf')
        best_action, best_child = None, None
        for a, child in self.children.items():
            q = 0 if child.visit_count == 0 else child.value_sum / child.visit_count
            u = C * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action, best_child = a, child
        return best_action, best_child

    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(-value)

class MCTS:
    def __init__(self, game, model, args,device):
        self.device=device
        self.game = game
        self.model = model
        self.args = args

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, state)
        # initial policy + noise
        board = self.game.change_perspective(state)
        encoded = self.game.get_encoded_state(board)
        x = torch.tensor(encoded, dtype=torch.float32, device=self.device).unsqueeze(0)
        policy_logits, value = self.model(x)
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        # mask
        valid = self.game.get_valid_moves(state)
        policy = policy * valid
        policy = policy / np.sum(policy)
        # dirichlet noise
        noise = np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        policy = (1 - self.args['epsilon']) * policy + self.args['epsilon'] * noise
        valid  = self.game.get_valid_moves(state)
        policy = policy * valid
        policy = policy / np.sum(policy)
        root.expand(policy)

        for _ in range(self.args['num_searches']):
            node = root
            # selection & expansion
            while node.is_expanded():
                a, node = node.select(self.args['C'])
            v, done = self.game.get_value_and_terminated(node.state)
            if not done:
                board = self.game.change_perspective(node.state)
                enc = self.game.get_encoded_state(board)
                enc_tensor = torch.tensor(enc, dtype=torch.float32, device=self.device).unsqueeze(0)
                p_logits, v_tensor = self.model(enc_tensor)
                p = F.softmax(p_logits, dim=1).squeeze(0).cpu().numpy()
                p = p * self.game.get_valid_moves(node.state)
                p = p / np.sum(p)
                node.expand(p)
                v = v_tensor.item()
            node.backpropagate(v)

        # compute action probabilities
        visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)
        actions = list(root.children.keys())
        probs = np.zeros(self.game.action_size, dtype=np.float32)
        probs[actions] = visits / np.sum(visits)
        return probs

class AlphaZero:
    def __init__(self, model, optimizer, game, args,device):
        self.device = next(model.parameters()).device
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, model, args,self.device)

    def self_play(self):
        memory = []
        board = self.game.get_initial_state()
        print("⮕ Starting new game on device:", self.device)
        print("   Initial FEN:", board.fen(), "\n")
        player_turn = board.turn
        while True:
            state = board.copy()
            probs = self.mcts.search(state)

            # pick move
            action = np.random.choice(self.game.action_size, p=probs)

            # reconstruct the Move object (including pawn promotions)
            from_sq = action // 64
            to_sq   = action % 64
            move = chess.Move(from_sq, to_sq)
            if move not in state.legal_moves:
                # try queen promo if it’s a pawn promotion
                move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

            print(f"⮕ Player {'White' if player_turn else 'Black'} plays {move.uci()}")
            print("   FEN before move:", state.fen())

            board = self.game.get_next_state(board, action)

            print("   FEN after  move:", board.fen(), "\n")

            memory.append((state, probs, player_turn))
            value, done = self.game.get_value_and_terminated(board)
            if done:
                return [
                    (self.game.get_encoded_state(self.game.change_perspective(s)),
                     p,
                     value if pt == player_turn else -value)
                    for (s,p,pt) in memory
                ]
            player_turn = not player_turn

    def train(self, memory):
        random.shuffle(memory)
        for i in range(0, len(memory), self.args['batch_size']):
            batch = memory[i:i+self.args['batch_size']]
            states, pis, vs = zip(*batch)
            states = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
            pis = torch.tensor(np.stack(pis), dtype=torch.float32, device=self.device)
            vs = torch.tensor(vs, dtype=torch.float32, device=self.device).unsqueeze(1)
            self.optimizer.zero_grad()
            logits, val = self.model(states)
            loss_p = -torch.mean(torch.sum(pis * F.log_softmax(logits, dim=1), dim=1))
            loss_v = F.mse_loss(val, vs)
            (loss_p + loss_v).backward()
            self.optimizer.step()

    def learn(self):
        for it in range(self.args['num_iterations']):
            memory = []
            for _ in range(self.args['num_selfplay']):
                memory.extend(self.self_play())
            self.model.train()
            self.train(memory)
            # save checkpoint
            torch.save(self.model.state_dict(), f"az_model_{it}.pt")

if __name__ == '__main__':
    game = Chess960Game()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game).to(device)
    checkpoint = torch.load('/content/model_parameters.pth', map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    args = {
        'C':              1.4,
        'num_searches':   5,   # ← very few rollouts per move
        'epsilon':       0.25,
        'dirichlet_alpha':0.03,
        'batch_size':    64,   # ↓ smaller batch, less training work
        'num_iterations': 1,   # ← only one cycle
        'num_selfplay':   1    # ← only one self‐play game
    }
    az = AlphaZero(model, optimizer, game, args, device)
    az.learn()

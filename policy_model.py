import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
'''
Model which learns given a FEN data input to predict a move
The better this is the less we need to do self play. 
FEN DATA INPUT looks like r4r1k/1p4pp/3Pp3/1pp4q/3p1p2/P4P2/1P2QBPP/1R3RK1 b - - 0 23 h5d5
'''
def Get_Training_Data(chess_file: str):
    # reads in the chess file
    GAMES = []
    current_game = []
    with open(chess_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            record = line
            # new game starts when we see "... 0 1"
            if len(parts) >= 3 and parts[-3] == "0" and parts[-2] == "1":
                if current_game:
                    GAMES.append(current_game)
                current_game = [record]
            else:
                current_game.append(record)
        if current_game:
            GAMES.append(current_game)

    # keep only deep enough games
    GAMES = [g for g in GAMES if len(g) >= 15]
    # drop any game with a malformed line (<7 tokens)
    GAMES = [g for g in GAMES if all(len(r.split()) >= 7 for r in g)]
    # first game looked weird so I dropped
    if GAMES:
        GAMES = GAMES[1:]
    return GAMES

def encode_fen(fen: str) -> np.ndarray:
    """
    Turn a 6-token FEN into an (18,8,8) float32 array:
     - 12 piece planes
     - 1 side-to-move
     - 4 castling-rights planes
     - 1 en-passant plane
     input tensors will be 18x8x8
    """
    board = chess.Board(fen)
    X = np.zeros((18, 8, 8), dtype=np.float32)
    # pieces
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            plane = (p.piece_type - 1) + (0 if p.color else 6)
            r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
            X[plane, r, c] = 1
    # side to move
    X[12, :, :] = 1 if board.turn == chess.WHITE else 0
    # castling rights
    rights = {
        'K': chess.BB_H1,
        'Q': chess.BB_A1,
        'k': chess.BB_H8,
        'q': chess.BB_A8,
    }
    for i, bb in enumerate(rights.values(), start=13):
        X[i, :, :] = 1 if (board.castling_rights & bb) else 0
    # en passant
    if board.ep_square is not None:
        r, c = 7 - chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        X[17, r, c] = 1
    return X

def move_to_index(move_uci: str) -> int:
    #Moves are going to look like a tensor with 4096 output cells.
    #these represent some square to some other square movement
    m = chess.Move.from_uci(move_uci)
    return m.from_square * 64 + m.to_square

def create_training_tensors(chess_file: str):
    """
    Returns:
      X_tensor: FloatTensor of shape (N,18,8,8)
      y_tensor: LongTensor of shape (N,)
    """
    games = Get_Training_Data(chess_file)
    X_list, y_list = [], []
    for game in games:
        for rec in game:
            parts = rec.split()
            fen  = " ".join(parts[:6])
            mv   = parts[6]
            X_list.append(encode_fen(fen))
            y_list.append(move_to_index(mv))
    X_np = np.stack(X_list)                      # (N,18,8,8)
    y_np = np.array(y_list, dtype=np.int64)      # (N,)
    return torch.from_numpy(X_np), torch.from_numpy(y_np)

class ChessDataset(Dataset):
    def __init__(self, chess_file: str):
        self.X, self.y = create_training_tensors(chess_file)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
#RESNET MODEL MUST BE THE SAME AS POLICY PLUGGED INTO ALPHA ZERO
class ResBlock(nn.Module):
    def __init__(self, num_hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(num_hidden)
    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)

class ChessPolicyNet(nn.Module):
    def __init__(self, num_input_planes=18, num_resBlocks=4, num_hidden=128):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(num_input_planes, num_hidden, 3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backBone = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_resBlocks)])
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64 * 64)
        )
    def forward(self, x):
        x = self.startBlock(x)
        for b in self.backBone:
            x = b(x)
        return self.policyHead(x)

def train(chess_file, batch_size, lr, epochs, device):
    # data loader
    ds = ChessDataset(chess_file)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # model / optimizer / loss
    model     = ChessPolicyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # training loop
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)                 # (B, 4096)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {ep}/{epochs}  —  Avg Loss: {avg_loss:.4f}")
    return model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    model=train(
        chess_file = '/content/chess960_training_data (3).txt', #change if file is different
        batch_size = 64,
        lr         = .001,
        epochs     = 10,
        device     = device
    )
    path = './models/policy_model.pth'
  # Save the model's state_dict
    torch.save(model.state_dict(), path)

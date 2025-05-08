# Made with help of Claude
# Need to have teacher model saved prior to running this !

import chess
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from policy_model import ChessPolicyNet, encode_fen
from AlphaZeroChess960 import Chess960Game
import torch.profiler

teacher_path = "/home/wf2322/chess-distill/models/4_az_model_30.pth" # change to saved teacher model path

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

class StudentModel(nn.Module):
    def __init__(self, game: Chess960Game, num_input_planes=18, num_resBlocks=2, num_hidden=64):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(num_input_planes, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backBone = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_resBlocks)])
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.board_size * game.board_size, game.action_size)
        )
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

class Distiller:
    def __init__(self, teacher_model, student_model, optimizer, game, device, temperature=2.0):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.optimizer = optimizer
        self.game = game
        self.device = device
        self.temperature = temperature
        
    def distillation_loss(self, student_policy, teacher_policy, student_value, teacher_value, alpha=0.5):
        # Policy distillation with temperature scaling
        soft_targets = F.softmax(teacher_policy / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_policy / self.temperature, dim=1)
        policy_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Value distillation
        value_loss = F.mse_loss(student_value, teacher_value)
        
        # Combined loss
        return alpha * policy_loss + (1 - alpha) * value_loss
        
    @torch.no_grad()
    def generate_teacher_outputs(self, states, fens):
        self.teacher_model.eval()
        
        # Process each position with the policy model
        policy_outputs = []
        for fen in fens:
            # Encode FEN for ChessPolicyNet
            encoded_fen = encode_fen(fen)
            x = torch.tensor(encoded_fen, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Get policy output
            policy_logits = self.teacher_model(x)
            policy_outputs.append(policy_logits.squeeze(0))
        
        # Stack all outputs
        teacher_policy = torch.stack(policy_outputs)
        
        # Create dummy value outputs (since ChessPolicyNet doesn't have a value head)
        teacher_value = torch.zeros((len(states), 1), device=self.device)
        
        return teacher_policy, teacher_value
    
    def train_batch(self, states, fens):
        # Get teacher outputs
        teacher_policy, teacher_value = self.generate_teacher_outputs(states, fens)
        
        # Forward pass with student
        self.student_model.train()
        self.optimizer.zero_grad()
        student_policy, student_value = self.student_model(torch.tensor(states, dtype=torch.float32, device=self.device))
        
        # Calculate distillation loss
        loss = self.distillation_loss(student_policy, teacher_policy, student_value, teacher_value)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def distill(self, dataset, batch_size=64, epochs=10, profile=False):
        # Setup profiler
        profile_dir = "tbprofile/"
        
        # Create profiler schedule
        schedule = torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1
        )
        
        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            
            # Shuffle dataset
            indices = torch.randperm(len(dataset))
            
            # Profile first epoch to get FLOPs and model statistics
            if epoch == 0 and profile:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=schedule,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
                    record_shapes=True,
                    profile_memory=True,
                    with_flops=True,
                    with_stack=True
                ) as prof:
                    # Process a few batches with profiling
                    for i in range(0, min(5 * batch_size, len(dataset)), batch_size):
                        batch_indices = indices[i:i+batch_size]
                        states = [dataset[idx][0] for idx in batch_indices]
                        fens = [dataset[idx][1] for idx in batch_indices]
                        states = np.stack(states)
                        
                        loss = self.train_batch(states, fens)
                        total_loss += loss
                        batches += 1
                        
                        prof.step()
                
                
                # Print profiler results
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
                flops_table = prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20)
                print(flops_table)
                
                # Process remaining batches without profiling
                for i in range(5 * batch_size, len(dataset), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    states = [dataset[idx][0] for idx in batch_indices]
                    fens = [dataset[idx][1] for idx in batch_indices]
                    states = np.stack(states)
                    
                    loss = self.train_batch(states, fens)
                    total_loss += loss
                    batches += 1
            else:
                # Regular training without profiling
                for i in range(0, len(dataset), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    states = [dataset[idx][0] for idx in batch_indices]
                    fens = [dataset[idx][1] for idx in batch_indices]
                    states = np.stack(states)
                    
                    loss = self.train_batch(states, fens)
                    total_loss += loss
                    batches += 1
            
            avg_loss = total_loss / batches
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
          
            # Save checkpoint
            checkpoint_path = f"student_model_epoch_{epoch+1}.pt"
            torch.save(self.student_model.state_dict(), checkpoint_path)
            
    

def profile_models(teacher_model, student_model, device):
    """Profile both models to compare FLOPs and parameters"""
    # Create sample input
    sample_input = torch.randn(1, 18, 8, 8, device=device)
    
    # Profile teacher model
    print("Profiling teacher model...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_flops=True
    ) as prof_teacher:
        # Forward pass with teacher
        teacher_model(sample_input)
    
    teacher_stats = prof_teacher.key_averages()
    
    # Profile student model
    print("Profiling student model...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_flops=True
    ) as prof_student:
        # Forward pass with student
        student_model(sample_input)
    
    student_stats = prof_student.key_averages()
    
    # Print comparison
    print("\nModel Comparison:")
    print(f"Teacher total FLOPs: {sum(k.flops for k in teacher_stats if k.flops > 0)}")
    print(f"Student total FLOPs: {sum(k.flops for k in student_stats if k.flops > 0)}")
    print(f"Reduction: {(1 - sum(k.flops for k in student_stats if k.flops > 0) / sum(k.flops for k in teacher_stats if k.flops > 0)) * 100:.2f}%")
    
    return teacher_stats, student_stats

def main():
    # Initialize game
    game = Chess960Game()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load teacher model (ChessPolicyNet)
    teacher_model = ChessPolicyNet().to(device)
    ckpt = torch.load(teacher_path, map_location=device)

    # drop the value‑head keys
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith("valueHead")}

    teacher_model.load_state_dict(ckpt)   # strict=True
    teacher_model.eval()  # Set to evaluation mode
    
    # Create student model (smaller)
    student_model = StudentModel(game, num_resBlocks=16, num_hidden=64).to(device)
    
    # Profile both models before training
    teacher_stats, student_stats = profile_models(teacher_model, student_model, device)
    
    # Initialize optimizer for student
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    
    # Create dataset from standard chess positions
    dataset = []
    
    # Generate standard chess positions
    board = chess.Board()  # Standard chess starting position
    for _ in range(1000):
        encoded = game.get_encoded_state(board)
        fen = board.fen()  # Store FEN for ChessPolicyNet
        dataset.append((encoded, fen))  # Store both encoded state and FEN
        
        # Make some random moves to get different positions
        if not board.is_game_over():
            moves = list(board.legal_moves)
            if moves:
                board.push(random.choice(moves))
    
    # Initialize distiller
    distiller = Distiller(teacher_model, student_model, optimizer, game, device)
    
    # Run distillation with profiling enabled
    distiller.distill(dataset, batch_size=64, epochs=10, profile=True)
    
    # Save final student model
    torch.save(student_model.state_dict(), "final_student_model.pt")
    print("Distillation complete!")

if __name__ == "__main__":
    main()

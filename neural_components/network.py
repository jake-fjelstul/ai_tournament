"""
network.py
----------
A small Policy-Value network that takes an encoded board state and outputs:
 
  policy : (100,) float  — probability distribution over all possible moves
  value  : (1,)   float  — predicted score margin (positive = winning)
 
Architecture:
  Shared body  : 3 linear layers with ReLU + dropout for regularization
  Policy head  : 1 linear layer → softmax
  Value head   : 1 linear layer → tanh (squishes output to [-1, 1] range,
                 we scale by MAX_MARGIN to get actual point estimates)
 
We use PyTorch since it's already installed on the tournament machine.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_INPUTS  = 706   # encode.py output size
N_MOVES   = 100   # collect.py MOVE_VOCAB size
MAX_MARGIN = 100.0  # normalise value output — scores rarely exceed 100 pts
 
# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
 
class PolicyValueNet(nn.Module):
    """
    Shared-body MLP with a policy head and a value head.
 
    Body:
      706 → 512 → 256 → 128  (each with ReLU + dropout)
 
    Policy head:
      128 → 100  (log-softmax for cross-entropy loss during training,
                  softmax during inference)
 
    Value head:
      128 → 64 → 1  (tanh output, scaled by MAX_MARGIN)
    """
 
    def __init__(self, dropout=0.3):
        super().__init__()
 
        # Shared body
        self.body = nn.Sequential(
            nn.Linear(N_INPUTS, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
 
        # Policy head — outputs log-probabilities over 100 moves
        self.policy_head = nn.Sequential(
            nn.Linear(128, N_MOVES),
        )
 
        # Value head — outputs a single score estimate
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),   # output in [-1, 1], multiply by MAX_MARGIN to get points
        )
 
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, 706)
 
        Returns
        -------
        policy_logits : (batch, 100)  — raw logits, apply softmax for probs
        value         : (batch, 1)   — score estimate in [-MAX_MARGIN, MAX_MARGIN]
        """
        shared  = self.body(x)
        policy  = self.policy_head(shared)          # raw logits
        value   = self.value_head(shared) * MAX_MARGIN  # scale to point range
        return policy, value
 
    def predict(self, state_vec: np.ndarray):
        """
        Convenience method for use during game play (single state, no gradients).
 
        Parameters
        ----------
        state_vec : np.ndarray of shape (706,)
 
        Returns
        -------
        policy_probs : np.ndarray (100,) — probability over moves, sums to 1
        value        : float             — estimated score margin
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)  # (1, 706)
            logits, val = self.forward(x)
            probs = F.softmax(logits, dim=1).squeeze(0).numpy()
            return probs, float(val.squeeze())
 
 
# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
 
def train(dataset_path: str, model_path: str, epochs: int = 20, lr: float = 1e-3,
          batch_size: int = 256, val_split: float = 0.1):
    """
    Train the PolicyValueNet on the collected dataset.
 
    Parameters
    ----------
    dataset_path : str   — path to dataset.npy created by collect.py
    model_path   : str   — where to save the trained weights (.pth file)
    epochs       : int   — number of training epochs
    lr           : float — learning rate
    batch_size   : int   — mini-batch size
    val_split    : float — fraction of data to use for validation
    """
    print(f"Loading dataset from {dataset_path} ...")
    data     = np.load(dataset_path, allow_pickle=True).item()
    states   = data['states'].astype(np.float32)    # (N, 706)
    moves    = data['moves'].astype(np.int64)        # (N,)
    outcomes = data['outcomes'].astype(np.float32)  # (N,) score margins
 
    N = len(states)
    print(f"Dataset: {N} turns loaded.")
 
    # Train/val split
    val_size   = int(N * val_split)
    train_size = N - val_size
    idx        = np.random.permutation(N)
    train_idx, val_idx = idx[:train_size], idx[train_size:]
 
    X_train = torch.tensor(states[train_idx])
    y_move_train = torch.tensor(moves[train_idx])
    y_val_train  = torch.tensor(outcomes[train_idx]).unsqueeze(1)
 
    X_val = torch.tensor(states[val_idx])
    y_move_val = torch.tensor(moves[val_idx])
    y_val_val  = torch.tensor(outcomes[val_idx]).unsqueeze(1)
 
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
 
    model = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
 
    best_val_loss = float('inf')
 
    for epoch in range(1, epochs + 1):
        model.train()
        train_policy_loss = 0.0
        train_value_loss  = 0.0
        n_batches = 0
 
        # Shuffle each epoch
        perm = torch.randperm(train_size)
        X_train      = X_train[perm]
        y_move_train = y_move_train[perm]
        y_val_train  = y_val_train[perm]
 
        for start in range(0, train_size, batch_size):
            end = min(start + batch_size, train_size)
 
            xb      = X_train[start:end].to(device)
            mb      = y_move_train[start:end].to(device)
            vb      = y_val_train[start:end].to(device)
 
            policy_logits, value = model(xb)
 
            # Policy loss: cross-entropy against the move the strong bot chose
            policy_loss = F.cross_entropy(policy_logits, mb)
 
            # Value loss: MSE against the score margin
            # Normalise targets to [-1, 1] to match tanh output before scaling
            value_loss = F.mse_loss(value, vb)
 
            # Combined loss — weight policy slightly higher since it's
            # what drives move selection during play
            loss = policy_loss + 0.5 * value_loss
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            train_policy_loss += policy_loss.item()
            train_value_loss  += value_loss.item()
            n_batches += 1
 
        scheduler.step()
 
        # Validation
        model.eval()
        with torch.no_grad():
            vp_logits, vv = model(X_val.to(device))
            val_policy_loss = F.cross_entropy(vp_logits, y_move_val.to(device)).item()
            val_value_loss  = F.mse_loss(vv, y_val_val.to(device)).item()
 
            # Policy accuracy — did we predict the same move the strong bot chose?
            predicted_moves = vp_logits.argmax(dim=1)
            accuracy = (predicted_moves == y_move_val.to(device)).float().mean().item()
 
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Policy Loss: {train_policy_loss/n_batches:.4f} | "
              f"Value Loss: {train_value_loss/n_batches:.4f} | "
              f"Val Policy: {val_policy_loss:.4f} | "
              f"Val Value: {val_value_loss:.4f} | "
              f"Move Accuracy: {accuracy*100:.1f}%")
 
        # Save best model
        val_loss = val_policy_loss + 0.5 * val_value_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  → Saved best model to {model_path}")
 
    print(f"\nTraining complete. Best model saved to {model_path}")
 
 
# ---------------------------------------------------------------------------
# Loading a saved model
# ---------------------------------------------------------------------------
 
def load_model(model_path: str) -> PolicyValueNet:
    """
    Load a saved PolicyValueNet from disk.
 
    Parameters
    ----------
    model_path : str — path to the .pth file saved during training
 
    Returns
    -------
    PolicyValueNet with weights loaded, in eval mode
    """
    model = PolicyValueNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model
 
 
# ---------------------------------------------------------------------------
# Entry point — run training directly
# ---------------------------------------------------------------------------
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset.npy')
    parser.add_argument('--model',   default='model_weights.pth')
    parser.add_argument('--epochs',  type=int,   default=20)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--batch',   type=int,   default=256)
    args = parser.parse_args()
 
    train(
        dataset_path=args.dataset,
        model_path=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
    )
 
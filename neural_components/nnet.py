"""
nnet.py
-------
Wraps our PolicyValueNet to match the AlphaZero NeuralNet interface.

The interface expects:
  - train(examples)     where examples = [(state, pi, v), ...]
  - predict(board)      returns (pi, v)
  - save_checkpoint()
  - load_checkpoint()

We adapt our dataset format to fit:
  - board  → encoded state vector (706,)
  - pi     → policy vector (100,) — move probabilities
  - v      → float value (score margin)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from .network import PolicyValueNet, N_INPUTS, N_MOVES, MAX_MARGIN
from .encode import encode_state


class NNetWrapper:
    """
    Wraps PolicyValueNet to match the AlphaZero NeuralNet interface.
    """

    def __init__(self, game=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = PolicyValueNet().to(self.device)

    def train(self, examples):
        """
        Train on a list of (state, pi, v) examples.

        Parameters
        ----------
        examples : list of (np.ndarray, np.ndarray, float)
            state : (706,)  encoded board state
            pi    : (100,)  policy vector (move probabilities)
            v     : float   value (score margin, will be normalised)
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Unpack examples
        states  = torch.tensor(
            np.array([e[0] for e in examples]), dtype=torch.float32
        ).to(self.device)

        target_pis = torch.tensor(
            np.array([e[1] for e in examples]), dtype=torch.float32
        ).to(self.device)

        target_vs = torch.tensor(
            np.array([e[2] for e in examples], dtype=np.float32) / MAX_MARGIN,
            dtype=torch.float32
        ).unsqueeze(1).to(self.device)
        target_vs = target_vs.clamp(-1.0, 1.0)

        # Train for a few epochs on this batch
        for epoch in range(10):
            optimizer.zero_grad()

            policy_logits, value = self.model(states)

            # Policy loss: cross entropy against target policy
            policy_loss = -torch.mean(
                torch.sum(target_pis * F.log_softmax(policy_logits, dim=1), dim=1)
            )

            # Value loss: MSE against target value
            value_loss = F.mse_loss(value, target_vs)

            loss = policy_loss + 0.5 * value_loss
            loss.backward()
            optimizer.step()

        print(f"  Training loss: {loss.item():.4f} "
              f"(policy={policy_loss.item():.4f}, value={value_loss.item():.4f})")

    def train_from_dataset(self, dataset_path: str, epochs: int = 20,
                           batch_size: int = 256, lr: float = 1e-3):
        """
        Train directly from a dataset.npy file created by collect.py.
        This is the main entry point for training from collected data.

        Parameters
        ----------
        dataset_path : str  — path to dataset.npy
        epochs       : int  — number of training epochs
        batch_size   : int  — mini-batch size
        lr           : float — learning rate
        """
        print(f"Loading dataset from {dataset_path} ...")
        data     = np.load(dataset_path, allow_pickle=True).item()
        states   = data['states'].astype(np.float32)    # (N, 706)
        moves    = data['moves'].astype(np.int64)        # (N,)
        outcomes = data['outcomes'].astype(np.float32)  # (N,) score margins

        N = len(states)
        print(f"Dataset: {N} turns loaded.")

        # Normalise outcomes to [-1, 1] to match the tanh value head output.
        # Without this, a margin of 50 gives MSE loss of 50^2 = 2500 which
        # completely drowns out the policy loss.
        outcomes = outcomes / MAX_MARGIN
        outcomes = np.clip(outcomes, -1.0, 1.0)

        # Convert move indices to one-hot policy vectors
        pis = np.zeros((N, N_MOVES), dtype=np.float32)
        pis[np.arange(N), moves] = 1.0   # one-hot — strong bot chose this move

        # Build examples in AlphaZero format
        examples = list(zip(states, pis, outcomes))

        # Train/val split
        np.random.shuffle(examples)
        val_size   = int(N * 0.1)
        train_data = examples[val_size:]
        val_data   = examples[:val_size]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            self.model.train()
            np.random.shuffle(train_data)

            total_policy_loss = 0.0
            total_value_loss  = 0.0
            n_batches = 0

            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]

                xb  = torch.tensor(np.array([e[0] for e in batch]),
                                   dtype=torch.float32).to(self.device)
                pib = torch.tensor(np.array([e[1] for e in batch]),
                                   dtype=torch.float32).to(self.device)
                vb  = torch.tensor(np.array([e[2] for e in batch]),
                                   dtype=torch.float32).unsqueeze(1).to(self.device)

                policy_logits, value = self.model(xb)

                policy_loss = -torch.mean(
                    torch.sum(pib * F.log_softmax(policy_logits, dim=1), dim=1)
                )
                value_loss = F.mse_loss(value, vb)
                loss       = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                xv  = torch.tensor(np.array([e[0] for e in val_data]),
                                   dtype=torch.float32).to(self.device)
                piv = torch.tensor(np.array([e[1] for e in val_data]),
                                   dtype=torch.float32).to(self.device)
                vv  = torch.tensor(np.array([e[2] for e in val_data]),
                                   dtype=torch.float32).unsqueeze(1).to(self.device)

                vp_logits, vval = self.model(xv)
                val_policy_loss = -torch.mean(
                    torch.sum(piv * F.log_softmax(vp_logits, dim=1), dim=1)
                ).item()
                val_value_loss = F.mse_loss(vval, vv).item()

                predicted_moves = vp_logits.argmax(dim=1)
                actual_moves    = piv.argmax(dim=1)
                accuracy = (predicted_moves == actual_moves).float().mean().item()

            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Policy: {total_policy_loss/n_batches:.4f} | "
                  f"Value: {total_value_loss/n_batches:.4f} | "
                  f"Val Policy: {val_policy_loss:.4f} | "
                  f"Val Value: {val_value_loss:.4f} | "
                  f"Move Accuracy: {accuracy*100:.1f}%")

            val_loss = val_policy_loss + 0.5 * val_value_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('checkpoints', 'best.pth')
                print(f"  → Saved best model")

        print(f"\nTraining complete.")

    def predict(self, board_or_state, board=None):
        """
        Parameters
        ----------
        board_or_state : np.ndarray of shape (706,) — encoded board state
        board          : Board object (optional) — used to mask invalid moves.
                         Always pass this during gameplay to avoid predicting
                         invalid moves which cause instant loss.

        Returns
        -------
        pi : np.ndarray (100,) — move probabilities, invalid moves zeroed out
        v  : float             — estimated score margin (normalised to [-1, 1])
        """
        if not isinstance(board_or_state, np.ndarray):
            state = np.zeros(N_INPUTS, dtype=np.float32)
        else:
            state = board_or_state

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits, v = self.model(x)
            pi = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Mask invalid moves — set their probability to zero then renormalise.
        # This prevents the agent from ever choosing a move that would cause
        # an instant loss due to invalidity.
        if board is not None:
            from .collect import move_to_index
            valid_moves = board.get_valid_moves(exclude_search=False)
            valid_indices = set(move_to_index(m) for m in valid_moves)

            mask = np.zeros(len(pi), dtype=np.float32)
            for idx in valid_indices:
                if 0 <= idx < len(pi):
                    mask[idx] = 1.0

            pi = pi * mask

            # Renormalise — if all valid moves got zero probability
            # (shouldn't happen but safety net) fall back to uniform over valid
            total = pi.sum()
            if total > 1e-8:
                pi = pi / total
            else:
                pi = mask / mask.sum()

        return pi, float(v.squeeze().cpu())

    def save_checkpoint(self, folder: str = 'checkpoints', filename: str = 'model.pth'):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, folder: str = 'checkpoints', filename: str = 'model.pth'):
        path = os.path.join(folder, filename)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Loaded model from {path}")
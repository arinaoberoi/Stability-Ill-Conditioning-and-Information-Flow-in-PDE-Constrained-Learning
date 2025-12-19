import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """
    Minimal network for PDE-constrained learning.
    Purposefully small to expose failure modes.
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

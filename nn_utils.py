"""
Small neural-network utilities and helper functions.
"""

from __future__ import annotations

import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden_dim: int = 256, depth: int = 2) -> nn.Sequential:
    """
    A generic MLP builder: given input/output dimensions, compose an MLP with
    repeated (Linear + ReLU) blocks and a final Linear output layer.
    """
    layers: list[nn.Module] = []  # Collect modules in a Python list first.
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
        d = hidden_dim
    # Final linear output layer (no ReLU): many heads (e.g., mean/value/reward) need an
    # unconstrained real-valued output and should not be clipped to non-negative values.
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)  # Pack layers into nn.Sequential for net(x) forward.


"""
Minimal MBPO components for Pendulum-v1.

We implement a small probabilistic dynamics ensemble that predicts:
- next_state (via delta-state prediction)
- reward

This is intentionally lightweight and designed to plug into the unified runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        out = self.net(x)
        delta = out[..., : self.state_dim]
        reward = out[..., self.state_dim :]
        next_state = state + delta
        return next_state, reward


class DynamicsEnsemble(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, ensemble_size: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList(
            [DynamicsModel(state_dim, action_dim, hidden_dim=hidden_dim) for _ in range(ensemble_size)]
        )

    @torch.no_grad()
    def predict(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next_state and reward using a randomly selected ensemble member per batch element.
        """
        b = state.shape[0]
        device = state.device
        idx = torch.randint(0, self.ensemble_size, (b,), device=device)
        next_states = torch.empty((b, self.state_dim), device=device, dtype=state.dtype)
        rewards = torch.empty((b, 1), device=device, dtype=state.dtype)
        for i, model in enumerate(self.models):
            mask = idx == i
            if not torch.any(mask):
                continue
            ns, r = model(state[mask], action[mask])
            next_states[mask] = ns
            rewards[mask] = r
        return next_states, rewards


@dataclass
class ModelTrainStats:
    loss: float
    mse_next_state: float
    mse_reward: float


def train_dynamics_ensemble(
    ensemble: DynamicsEnsemble,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> ModelTrainStats:
    state, action, reward, next_state, done = batch
    # done is ignored for Pendulum (no terminals except time limit)
    del done

    loss_total = 0.0
    mse_s_total = 0.0
    mse_r_total = 0.0
    for model in ensemble.models:
        pred_next, pred_reward = model(state, action)
        mse_s = F.mse_loss(pred_next, next_state)
        mse_r = F.mse_loss(pred_reward, reward)
        loss = mse_s + mse_r
        loss_total = loss_total + loss
        mse_s_total = mse_s_total + mse_s
        mse_r_total = mse_r_total + mse_r

    loss_total = loss_total / len(ensemble.models)
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    mse_s_total = mse_s_total / len(ensemble.models)
    mse_r_total = mse_r_total / len(ensemble.models)
    return ModelTrainStats(
        loss=float(loss_total.item()),
        mse_next_state=float(mse_s_total.item()),
        mse_reward=float(mse_r_total.item()),
    )


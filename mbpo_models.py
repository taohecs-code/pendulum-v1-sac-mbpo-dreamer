r"""
MBPO dynamics model.

1. Probabilistic dynamics ensemble: train an ensemble of models to predict environment dynamics.
2. Data augmentation: use the learned model to generate short synthetic rollouts and train an off-policy learner (SAC).
3. Model accuracy monitoring: compute MSE between predicted next state and the true next state in observation space.
4. Rollout horizon is a key experimental variable: moderate horizons can help; overly long horizons can accumulate error.

Implementation notes (this file):
- We model targets as \(y=[\Delta s, r]\) and predict a diagonal Gaussian: mean and log_std.
- We also predict a terminal probability (done_for_learning = terminated) via a Bernoulli head (logit).
- For stability we clamp log_std to [-20, 2] (same range used in SAC actor).
- The model is trained with Gaussian NLL, while still logging MSE as a metric.
- We train an ensemble (default 7 models) and select a subset (top-k, default 5) by lowest validation MSE.
- We log epistemic variance (disagreement across models) to monitor model uncertainty and potential rollout drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    """
    A single MLP dynamics model (one member of the ensemble).

    It learns a probabilistic transition model in low-dimensional observation space.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Probabilistic head (diagonal Gaussian): y = [delta_state, reward]
        self.gaussian_head = nn.Linear(hidden_dim, 2 * (state_dim + 1))

        # Terminal head (Bernoulli logit): predicts done_for_learning (= terminated).
        self.terminated_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns distribution parameters for y = [delta_state, reward] plus a terminal logit:
          mu:      (B, state_dim + 1)
          log_std: (B, state_dim + 1)
          terminated_logit: (B, 1)
        """
        x = torch.cat([state, action], dim=-1)
        h = self.trunk(x)
        out = self.gaussian_head(h)

        # Split the last-layer output into two halves:
        # - mean (mu)
        # - log_std (log of standard deviation)
        d = self.state_dim + 1 
        mu, log_std = out[..., :d], out[..., d:] 
        # Use the same clamp range as SAC actor for stability.
        log_std = torch.clamp(log_std, -20, 2)
        terminated_logit = self.terminated_head(h)
        return mu, log_std, terminated_logit

    @torch.no_grad()
    def mean_prediction(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deterministic prediction using the mean only (no sampling/no noise).

        This is typically used for evaluation/metrics (e.g., reporting MSE of next_state),
        because it is more stable and interpretable than using a stochastic sample.
        """
        mu, _, terminated_logit = self.forward(state, action)
        delta_mu = mu[..., : self.state_dim]
        reward_mu = mu[..., self.state_dim :]
        next_state_mu = state + delta_mu  # model delta-state, so we add it to the current state to get next_state.
        terminated_prob = torch.sigmoid(terminated_logit)
        return next_state_mu, reward_mu, terminated_prob

    @torch.no_grad()
    def sample_prediction(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample once from the learned diagonal-Gaussian dynamics model, producing a noisy prediction (s', r).

        This is used during MBPO rollouts to generate synthetic transitions and to inject model uncertainty.
        """
        mu, log_std, terminated_logit = self.forward(state, action)
        # Convert log_std -> std
        std = torch.exp(log_std) 

        # Sample epsilon ~ N(0, I)
        eps = torch.randn_like(std)

        # Reparameterization: y = mu + std * eps
        y = mu + std * eps
        delta = y[..., : self.state_dim]
        reward = y[..., self.state_dim :]
        next_state = state + delta
        terminated_prob = torch.sigmoid(terminated_logit)
        return next_state, reward, terminated_prob


class DynamicsEnsemble(nn.Module):
    """
    An ensemble of dynamics models.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 7,
        hidden_dim: int = 256,
        top_k_models: int = 5,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.top_k_models = top_k_models
        self.models = nn.ModuleList(
            [DynamicsModel(state_dim, action_dim, hidden_dim=hidden_dim) for _ in range(ensemble_size)]
        )
        # Avoid out-of-range indices when top_k_models > ensemble_size
        self.selected_model_indices: List[int] = list(range(min(top_k_models, ensemble_size)))

    @torch.no_grad()
    def predict(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next_state and reward using a randomly selected model from the selected subset per batch element,
        and sample from that model's Gaussian output (aleatoric uncertainty).
        """
        b = state.shape[0] # batch size
        device = state.device

        # The subset of selected models is updated during training (top-k by validation MSE).
        # During rollouts we randomly choose one selected model per batch element.
        selected = torch.as_tensor(self.selected_model_indices, device=device, dtype=torch.long)
        # Sample an index into the selected subset for each batch element.
        choice = torch.randint(0, selected.numel(), (b,), device=device)
        # Map subset index -> actual model index.
        idx = selected[choice]
        # Pre-allocate outputs, then fill them by model-specific masks below.
        next_states = torch.empty((b, self.state_dim), device=device, dtype=state.dtype)
        rewards = torch.empty((b, 1), device=device, dtype=state.dtype)
        terminated_probs = torch.empty((b, 1), device=device, dtype=state.dtype)
        
        for i, model in enumerate(self.models):
            mask = idx == i
            if not torch.any(mask):
                continue
            ns, r, term_prob = model.sample_prediction(state[mask], action[mask])
            next_states[mask] = ns
            rewards[mask] = r
            terminated_probs[mask] = term_prob
        terminated = (terminated_probs > 0.5).to(dtype=state.dtype)
        return next_states, rewards, terminated

    @torch.no_grad()
    def selection_mse(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """
        Compute per-model validation MSE (next_state + reward) to select top-k models.
        Returns: (ensemble_size,)
        """
        errs = []
        for model in self.models:
            ns_mu, r_mu, _ = model.mean_prediction(state, action)
            mse_s = F.mse_loss(ns_mu, next_state, reduction="mean")
            mse_r = F.mse_loss(r_mu, reward, reduction="mean")
            errs.append(mse_s + mse_r)
        return torch.stack(errs, dim=0)


@dataclass
class ModelTrainStats:
    # optimization objective
    nll: float
    loss: float  # keep alias name for runner compatibility
    # metrics (proposal)
    mse_next_state: float
    mse_reward: float
    # extra diagnostics
    avg_log_std: float
    epistemic_var_next_state: float
    epistemic_var_reward: float
    selected_model_indices: List[int]
    terminated_bce: float = 0.0
    terminated_acc: float = 0.0
    terminated_rate: float = 0.0


def train_dynamics_ensemble(
    ensemble: DynamicsEnsemble,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> ModelTrainStats:
    state, action, reward, next_state, done = batch

    # target y = [delta_state, reward] from replay buffer
    delta_target = next_state - state
    y = torch.cat([delta_target, reward], dim=-1)

    # Simple train/val split from the same batch:
    # - train split: optimize NLL
    # - val split: choose top-k models by lowest MSE
    b = state.shape[0]
    # Random permutation of indices 0..b-1.
    perm = torch.randperm(b, device=state.device)
    if b < 2:
        # Robustness: extremely small batches (e.g., b=1) would create an empty train or val split.
        # In that case, just use the same tiny set for both train and selection metrics.
        train_idx = perm
        val_idx = perm
    else:
        # Validation set size = 1/5 of batch (at least 1).
        val_b = max(1, b // 5)
        train_b = b - val_b
        # First part -> train, second part -> val.
        train_idx = perm[:train_b]
        val_idx = perm[train_b:]

    s_tr, a_tr, y_tr = state[train_idx], action[train_idx], y[train_idx]
    done_tr = done[train_idx]
    s_val, a_val = state[val_idx], action[val_idx]
    ns_val, r_val = next_state[val_idx], reward[val_idx]

    nll_sum = 0.0
    mse_s_sum = 0.0
    mse_r_sum = 0.0
    log_std_sum = 0.0
    term_bce_sum = 0.0
    term_acc_sum = 0.0

    # For epistemic variance: collect mean predictions across models on the full batch.
    mu_deltas = []
    mu_rewards = []

    for model in ensemble.models:
        mu, log_std, term_logit = model(s_tr, a_tr)  # (B, D), (B, D), (B, 1)
        std = torch.exp(log_std)
        # Gaussian NLL (diagonal), ignore constant term
        # nll = 0.5 * [ ((y-mu)/std)^2 + 2*log_std ]
        # - ((y - mu) / std)^2: squared error scaled by uncertainty
        # - 2*log_std: prevents the model from inflating std to avoid error
        # - +1e-8: avoids division by zero
        # Note: nll is still (B, D) before reduction.

        nll = 0.5 * (((y_tr - mu) / (std + 1e-8)).pow(2) + 2.0 * log_std)
        nll = nll.mean()
        nll_sum = nll_sum + nll

        term_bce = F.binary_cross_entropy_with_logits(term_logit, done_tr)
        term_bce_sum = term_bce_sum + term_bce
        term_prob = torch.sigmoid(term_logit)
        term_acc = ((term_prob > 0.5).to(dtype=done_tr.dtype) == (done_tr > 0.5).to(dtype=done_tr.dtype)).float().mean()
        term_acc_sum = term_acc_sum + term_acc

        # metrics on full batch (use mean prediction)
        # compute MSE using the mean prediction mu (deterministic),
        # and compare against ground-truth next_state/reward stored in the replay buffer.
        mu_full, log_std_full, _ = model(state, action)
        delta_mu = mu_full[..., : ensemble.state_dim]
        r_mu = mu_full[..., ensemble.state_dim :]
        ns_mu = state + delta_mu
        mse_s_sum = mse_s_sum + F.mse_loss(ns_mu, next_state, reduction="mean")
        mse_r_sum = mse_r_sum + F.mse_loss(r_mu, reward, reduction="mean")
        log_std_sum = log_std_sum + log_std_full.mean()

        mu_deltas.append(delta_mu)
        mu_rewards.append(r_mu)

    nll_avg = nll_sum / len(ensemble.models)
    term_bce_avg = term_bce_sum / len(ensemble.models)
    term_acc_avg = term_acc_sum / len(ensemble.models)
    term_rate = float(done.mean().item())

    total_loss = nll_avg + term_bce_avg
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    mse_s_avg = mse_s_sum / len(ensemble.models)
    mse_r_avg = mse_r_sum / len(ensemble.models)
    # Global diagnostic: average predicted log_std over models and batch.
    # This is a coarse summary (it can hide per-model differences), but is useful for spotting
    # pathological uncertainty inflation/deflation.
    avg_log_std = log_std_sum / len(ensemble.models)

    # model selection by validation MSE (next_state + reward)
    with torch.no_grad():
        val_errs = ensemble.selection_mse(s_val, a_val, ns_val, r_val)  # (E,) from mini batch validation set(1/5 of batch)
        k = min(ensemble.top_k_models, ensemble.ensemble_size)
        selected = torch.topk(val_errs, k=k, largest=False).indices.tolist()
        ensemble.selected_model_indices = selected

        # Epistemic variance across models (disagreement in mean predictions).
        delta_stack = torch.stack(mu_deltas, dim=0)  # (E, B, state_dim)
        reward_stack = torch.stack(mu_rewards, dim=0)  # (E, B, 1)
        epistemic_var_next = delta_stack.var(dim=0, unbiased=False).mean()
        epistemic_var_reward = reward_stack.var(dim=0, unbiased=False).mean()

    return ModelTrainStats(
        nll=float(nll_avg.item()),
        loss=float(total_loss.item()),
        mse_next_state=float(mse_s_avg.item()),
        mse_reward=float(mse_r_avg.item()),
        avg_log_std=float(avg_log_std.item()),
        epistemic_var_next_state=float(epistemic_var_next.item()),
        epistemic_var_reward=float(epistemic_var_reward.item()),
        selected_model_indices=list(ensemble.selected_model_indices),
        terminated_bce=float(term_bce_avg.item()),
        terminated_acc=float(term_acc_avg.item()),
        terminated_rate=float(term_rate),
    )


r"""
MBPO dynamics model.

1. Probabilistic dynamics ensemble: train an ensemble of models to predict environment dynamics.
2. Data augmentation: use the learned model to generate short synthetic rollouts and train an off-policy learner (SAC).
3. Model accuracy monitoring: compute MSE between predicted next state and the true next state in observation space.
4. Rollout horizon is a key experimental variable: moderate horizons can help; overly long horizons can accumulate error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    """
    A single MLP dynamics model (one member of the ensemble) to create rollouts.
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

        # reward(1) and delta state(state dimensions)
        # mean and log_std
        self.gaussian_head = nn.Linear(hidden_dim, 2 * (state_dim + 1))

        # Terminal head (Bernoulli logit): predicts done_for_learning (= terminated).
        # dimension is 1.
        self.terminated_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        # (B, state_dim + action_dim)
        x = torch.cat([state, action], dim=-1) 

        # network output
        h = self.trunk(x) 

        # reward(1) and delta state(state dimensions) with mean and log_std
        out = self.gaussian_head(h) 
        d = self.state_dim + 1 
        mu, log_std = out[..., :d], out[..., d:] 
        log_std = torch.clamp(log_std, -20, 2)

        # terminal logit for binary classification(terminated or not)
        terminated_logit = self.terminated_head(h)
        return mu, log_std, terminated_logit

    @torch.no_grad()
    def mean_prediction(self, state: torch.Tensor, action: torch.Tensor):
        """
        Deterministic prediction using the mean only (no sampling/no noise).
        """
        mu, _, terminated_logit = self.forward(state, action)
        delta_mu = mu[..., : self.state_dim]
        reward_mu = mu[..., self.state_dim :]
        next_state_mu = state + delta_mu 
        terminated_prob = torch.sigmoid(terminated_logit)
        return next_state_mu, reward_mu, terminated_prob

    @torch.no_grad()
    def sample_prediction(self, state: torch.Tensor, action: torch.Tensor):
        """
        Sample once from the learned dynamics model, producing a next state and reward.
        This is used during MBPO rollouts to generate synthetic transitions and to inject model uncertainty.
        """

        mu, log_std, terminated_logit = self.forward(state, action)
       
        std = torch.exp(log_std) 

        # Sample epsilon ~ N(0, I)
        # add noise to the mean
        eps = torch.randn_like(std)

        # Reparameterization
        y = mu + std * eps
        delta = y[..., : self.state_dim]
        reward = y[..., self.state_dim :]
        next_state = state + delta
        terminated_prob = torch.sigmoid(terminated_logit)
        return next_state, reward, terminated_prob


class DynamicsEnsemble(nn.Module):
    """
    An ensemble of dynamics models.
    Select the top-k models by validation MSE.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 7,
        hidden_dim: int = 256,
        top_k_models: int = 5,
        normalize_inputs_targets: bool = True,
        normalizer_eps: float = 1e-6,
        normalizer_momentum: float = 0.01,
    ):

        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.top_k_models = top_k_models
        self.normalize_inputs_targets = bool(normalize_inputs_targets)
        self.normalizer_eps = float(normalizer_eps)
        self.normalizer_momentum = float(normalizer_momentum)
        self.models = nn.ModuleList(
            [DynamicsModel(state_dim, action_dim, hidden_dim=hidden_dim) for _ in range(ensemble_size)]
        )

        self.selected_model_indices: List[int] = list(range(min(top_k_models, ensemble_size)))
        self.register_buffer("input_mean", torch.zeros(state_dim + action_dim))
        self.register_buffer("input_std", torch.ones(state_dim + action_dim))
        self.register_buffer("target_mean", torch.zeros(state_dim + 1))
        self.register_buffer("target_std", torch.ones(state_dim + 1))
        self.register_buffer("_normalizer_ready", torch.tensor(False))

    # ================================
    # Normalization
    # ================================
    # Important data preprocessing: normalize the inputs for the dynamics model(network input) for stability.
    # 1. update_normalizer: memorize the mean and std of the real data(state, action, target), and use EMA for updating history.
    # 2. _normalize_inputs:use Z-score normalization to normalize the inputs for the dynamics model(network input).
    # 3. _normalize_target: use Z-score normalization to normalize the targets for the dynamics model(network output).
    # 4. _denormalize_target: use Z-score denormalization to denormalize the targets for the dynamics model(network output).

    @torch.no_grad()
    def update_normalizer(self, state: torch.Tensor, action: torch.Tensor, target: torch.Tensor):

        if not self.normalize_inputs_targets:
            return
        x = torch.cat([state, action], dim=-1)
        x_mean = x.mean(dim=0)
        x_std = x.std(dim=0, unbiased=False).clamp_min(self.normalizer_eps)
        y_mean = target.mean(dim=0)
        y_std = target.std(dim=0, unbiased=False).clamp_min(self.normalizer_eps)

        if not bool(self._normalizer_ready.item()):
            self.input_mean.copy_(x_mean)
            self.input_std.copy_(x_std)
            self.target_mean.copy_(y_mean)
            self.target_std.copy_(y_std)
            self._normalizer_ready.fill_(True)
            return

        m = self.normalizer_momentum
        self.input_mean.mul_(1.0 - m).add_(m * x_mean)
        self.input_std.mul_(1.0 - m).add_(m * x_std)
        self.target_mean.mul_(1.0 - m).add_(m * y_mean)
        self.target_std.mul_(1.0 - m).add_(m * y_std)

    def _normalize_inputs(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.normalize_inputs_targets:
            return state, action
        x = torch.cat([state, action], dim=-1)
        x_norm = (x - self.input_mean) / self.input_std.clamp_min(self.normalizer_eps)
        return x_norm[..., : self.state_dim], x_norm[..., self.state_dim :]

    def _normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        if not self.normalize_inputs_targets:
            return target
        return (target - self.target_mean) / self.target_std.clamp_min(self.normalizer_eps)

    def _denormalize_target(self, target_norm: torch.Tensor) -> torch.Tensor:
        if not self.normalize_inputs_targets:
            return target_norm
        return target_norm * self.target_std + self.target_mean



    @torch.no_grad()
    def predict(self, state: torch.Tensor, action: torch.Tensor):
        """
        Predict next_state and reward using a randomly selected model from the selected subset per batch element(*rollout*),
        and sample from that model's Gaussian output (aleatoric uncertainty).
        """
        b = state.shape[0] # batch size
        device = state.device

        # Choose top k models by validation MSE.
        selected = torch.as_tensor(self.selected_model_indices, device=device, dtype=torch.long) 
        # Randomly select one model from the selected subset.
        choice = torch.randint(0, selected.numel(), (b,), device=device)
        idx = selected[choice]

        next_states = torch.empty((b, self.state_dim), device=device, dtype=state.dtype)
        rewards = torch.empty((b, 1), device=device, dtype=state.dtype)
        terminated_probs = torch.empty((b, 1), device=device, dtype=state.dtype)
        
        for i, model in enumerate(self.models):
            mask = idx == i
            if not torch.any(mask):
                continue
            s_in, a_in = self._normalize_inputs(state[mask], action[mask])
            mu, log_std, term_logit = model(s_in, a_in)
            std = torch.exp(log_std)
            y_norm = mu + std * torch.randn_like(std)
            y = self._denormalize_target(y_norm)
            delta = y[..., : self.state_dim]
            r = y[..., self.state_dim :]
            ns = state[mask] + delta
            term_prob = torch.sigmoid(term_logit)
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
            s_in, a_in = self._normalize_inputs(state, action)
            mu, _, _ = model(s_in, a_in)
            mu = self._denormalize_target(mu)
            delta_mu = mu[..., : self.state_dim]
            r_mu = mu[..., self.state_dim :]
            ns_mu = state + delta_mu

            mse_s = F.mse_loss(ns_mu, next_state, reduction="mean")
            mse_r = F.mse_loss(r_mu, reward, reduction="mean")
            errs.append(mse_s + mse_r)

        return torch.stack(errs, dim=0)



def train_dynamics_ensemble(
    ensemble: DynamicsEnsemble,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
):

    """
    Trains the ensemble of dynamics models for one gradient step.

    This function performs the following operations:
    1. Computes the target as the state difference (next_state - state) and reward.
    2. Updates the input/target normalizers using the current batch.
    3. Splits the batch into training and validation sets.
    4. Computes the Negative Log-Likelihood (NLL) loss for the Gaussian predictions
       and Binary Cross-Entropy (BCE) loss for the termination predictions.
    5. Performs a backward pass and updates the ensemble's weights.
    6. Evaluates the models on the validation set to select the top-K performing models.

    Args:
        ensemble: The dynamics ensemble model to be trained.
        batch: A tuple of tensors (state, action, reward, next_state, done) from the replay buffer.
        optimizer: The PyTorch optimizer used to update the ensemble's parameters.

    Returns:
        ModelTrainStats: A dataclass containing training metrics and diagnostics.
    """

    # Unpack the batch from the replay buffer.
    state, action, reward, next_state, done = batch

    # Form the input and target for the dynamics model.
    delta_target = next_state - state
    y = torch.cat([delta_target, reward], dim=-1)

    # Update the normalizer.
    ensemble.update_normalizer(state, action, y)

    # Randomly permute the indices.
    b = state.shape[0]
    perm = torch.randperm(b, device=state.device)

    # Split the batch into train and validation sets.
    if b < 2:
        train_idx = perm
        val_idx = perm
    else:
        val_b = max(1, b // 5)
        train_b = b - val_b
        train_idx = perm[:train_b]
        val_idx = perm[train_b:]

    s_tr, a_tr, y_tr = state[train_idx], action[train_idx], y[train_idx]
    s_tr_in, a_tr_in = ensemble._normalize_inputs(s_tr, a_tr)
    y_tr_norm = ensemble._normalize_target(y_tr)
    done_tr = done[train_idx]
    s_val, a_val = state[val_idx], action[val_idx]
    ns_val, r_val = next_state[val_idx], reward[val_idx]

    # Initialize the loss and metrics for all ensemble models.
    
    # Negative Log-Likelihood loss for the Gaussian predictions
    nll_sum = 0.0 

    # Mean Squared Error loss for the next state and reward predictions.
    mse_s_sum = 0.0
    mse_r_sum = 0.0
    log_std_sum = 0.0

    # Termination Binary Cross-Entropy loss and accuracy.
    term_bce_sum = 0.0
    term_acc_sum = 0.0

    # For epistemic uncertainty: collect mean predictions across models.
    mu_deltas = []
    mu_rewards = []

    for model in ensemble.models:
        mu, log_std, term_logit = model(s_tr_in, a_tr_in)  # (B, D), (B, D), (B, 1)
        std = torch.exp(log_std)


        # Gaussian NLL (diagonal), ignore constant term
        # nll = 0.5 * [ ((y-mu)/std)^2 + 2*log_std ]
        # - ((y - mu) / std)^2: squared error scaled by uncertainty
        # - 2*log_std: prevents the model from inflating std to avoid error
        # - +1e-8: avoids division by zero
        # Note: nll is still (B, D) before reduction.
        nll = 0.5 * (((y_tr_norm - mu) / (std + 1e-8)).pow(2) + 2.0 * log_std)
        nll = nll.mean()
        nll_sum = nll_sum + nll

        # Termination Binary Cross-Entropy loss and accuracy.
        term_bce = F.binary_cross_entropy_with_logits(term_logit, done_tr)
        term_bce_sum = term_bce_sum + term_bce
        term_prob = torch.sigmoid(term_logit)
        term_acc = ((term_prob > 0.5).to(dtype=done_tr.dtype) == (done_tr > 0.5).to(dtype=done_tr.dtype)).float().mean()
        term_acc_sum = term_acc_sum + term_acc

        # ================================
        # Calculate info across batch.
        # ================================
        with torch.no_grad():
            s_full_in, a_full_in = ensemble._normalize_inputs(state, action)
            mu_full, log_std_full, _ = model(s_full_in, a_full_in)
            mu_full = ensemble._denormalize_target(mu_full)
            delta_mu = mu_full[..., : ensemble.state_dim]
            r_mu = mu_full[..., ensemble.state_dim :]
            ns_mu = state + delta_mu
            # calculate and accumulate metrics.
            mse_s_sum = mse_s_sum + F.mse_loss(ns_mu, next_state, reduction="mean")
            mse_r_sum = mse_r_sum + F.mse_loss(r_mu, reward, reduction="mean")
            log_std_sum = log_std_sum + log_std_full.mean()

            mu_deltas.append(delta_mu)
            mu_rewards.append(r_mu)

    # ================================
    # Train the ensemble of dynamics models using NLL loss and termination loss.
    # ================================
    nll_avg = nll_sum / len(ensemble.models)
    term_bce_avg = term_bce_sum / len(ensemble.models)
    term_acc_avg = term_acc_sum / len(ensemble.models)
    term_rate = float(done.mean().item())

    total_loss = nll_avg + term_bce_avg

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


    # ================================
    # Calculate average metrics across models.
    # ================================
    mse_s_avg = mse_s_sum / len(ensemble.models)
    mse_r_avg = mse_r_sum / len(ensemble.models)
    avg_log_std = log_std_sum / len(ensemble.models)

    # ================================
    # Top-k model selection by validation MSE (next_state + reward).
    # ================================
    with torch.no_grad():
        val_errs = ensemble.selection_mse(s_val, a_val, ns_val, r_val)  
        k = min(ensemble.top_k_models, ensemble.ensemble_size)
        selected = torch.topk(val_errs, k=k, largest=False).indices.tolist()
        ensemble.selected_model_indices = selected

        # Epistemic variance across models (disagreement in mean predictions).
        delta_stack = torch.stack(mu_deltas, dim=0)  
        reward_stack = torch.stack(mu_rewards, dim=0)  
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

@dataclass
class ModelTrainStats:
    nll: float
    loss: float  
    mse_next_state: float
    mse_reward: float
    avg_log_std: float
    epistemic_var_next_state: float
    epistemic_var_reward: float
    selected_model_indices: List[int]
    terminated_bce: float = 0.0
    terminated_acc: float = 0.0
    terminated_rate: float = 0.0

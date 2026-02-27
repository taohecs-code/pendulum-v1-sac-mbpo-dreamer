"""
Simplified Dreamer-style world model for low-dimensional observations.

1. Replace CNN with MLP: Pendulum-v1 has a 3D state, so we drop Dreamer's high-dimensional CNNs and use MLPs for encoder/decoder.
2. RSSM (Recurrent State-Space Model): models stochastic action-conditioned transitions in latent space for multi-step imagination.
3. Probabilistic decoder / reward head: the world model is trained with diagonal Gaussian negative log-likelihood (NLL). For monitoring we also log MSE.
   - If `DreamerConfig.learn_std=True`, the decoder and reward head predict both mean and log_std.
   - If `learn_std=False`, std is fixed via `obs_std` / `reward_std`.
   - Predicted log_std is clamped to `[log_std_min, log_std_max]` for numerical stability.
4. Continuation head: predicts episode continuation probability p(cont_t=1 | h_t, s_t) (used as a learned discount factor and trained with BCE in the agent).
5. KL regularization: RSSM posterior/prior are regularized with KL(q||p), with optional KL balancing (`kl_balance`) and free nats (`free_nats`).

This module contains the components that make up the "world model" in Dreamer:
- DreamerConfig: hyperparameters shared by model/agent
- RSSM: recurrent state-space model (latent dynamics core)
- WorldModel: encoder + RSSM + decoder + reward head + continuation head, trained via probabilistic reconstruction + KL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_utils import mlp


@dataclass
class DreamerConfig:

    latent_dim: int = 32
    deter_dim: int = 64
    hidden_dim: int = 256
    horizon: int = 15

    model_lr: float = 1e-3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # World model loss weights / stabilizers
    kl_beta: float = 1.0
    continuation_beta: float = 1.0

    # KL stabilization tricks commonly used in Dreamer implementations.
    free_nats: float = 3.0
    kl_balance: float = 0.8  # 1.0 => posterior-only grads, 0.0 => prior-only grads

    # Likelihood / distribution settings.
    # If learn_std=True, decoder/reward head predict log_std; otherwise fixed stds are used.
    learn_std: bool = True
    obs_std: float = 0.1
    reward_std: float = 1.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    target_tau: float = 0.01

    # Lambda-return mixing coefficient (TD(lambda) style).
    lambda_: float = 0.95

    # Actor regularization (optional).
    actor_entropy_coef: float = 0.0

    gamma: float = 0.99


class RSSM(nn.Module):
    """
    Recurrent State Space Model (RSSM) for Dreamer-style agent.

    RSSM is the "dynamics core" of the world model: it performs multi-step, memory-based latent transitions.

    What RSSM maintains (the latent state):
    - deter (deterministic state): the GRU hidden state that stores history/memory. In the Dreamer paper this is typically
      written as h_t.
    - stoch (stochastic state): a stochastic latent variable sampled from a diagonal Gaussian. In the paper this is
      typically written as s_t.
    The complete latent state is therefore (deter, stoch) ≈ (h_t, s_t).

    What RSSM provides (the core functions/heads):
    - recurrent update (inp + GRUCell): updates deter using previous latent state and action
    - prior head: predicts parameters of p(stoch_t | deter_t) = N(mean, std^2)
    - posterior head: predicts parameters of q(stoch_t | deter_t, emb_t) = N(mean, std^2)
      where emb_t is the observation embedding e_t = encoder(o_t)

    Why there are both prior and posterior:
    - imagination / rollout: there is no real observation, so we can only use the prior p(stoch_t | deter_t)
    - training with real data: we do have o_t, so we can use the posterior q(stoch_t | deter_t, emb_t) to "correct"
      the stochastic latent (this keeps the latent trajectory anchored to observations)

    Notation mapping to the Dreamer paper:
    - deter ↔ h_t (deterministic RNN hidden state)
    - stoch ↔ s_t (stochastic latent state)
    - action ↔ a_{t-1} or a_t (depending on indexing convention)
    - obs[:, i] ↔ o_t (observation)
    - emb = encoder(obs[:, i]) ↔ e_t = f(o_t) (observation embedding)
    """

    def __init__(self, action_dim: int, deter_dim: int, stoch_dim: int, hidden_dim: int = 256):

        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        # Recurrent update uses previous latent state and action.
        # concatenate (deter_{t-1}, stoch_{t-1}, action_{t-1}) as input.
        self.inp = nn.Linear(deter_dim + stoch_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # prior: p(stoch_t | deter_t) -> (mean, log_std)
        self.prior = mlp(deter_dim, 2 * stoch_dim, hidden_dim=hidden_dim, depth=1)

        # posterior: q(stoch_t | deter_t, emb_t) -> (mean, log_std)
        self.posterior = mlp(deter_dim + stoch_dim, 2 * stoch_dim, hidden_dim=hidden_dim, depth=1)

    def init_state(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the RSSM latent state for the first timestep of a batch of sequences.
        """

        deter = torch.zeros((batch, self.deter_dim), device=device)
        stoch = torch.zeros((batch, self.stoch_dim), device=device)
        return deter, stoch

    def prior_step(
        self, deter: torch.Tensor, stoch: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Recurrent transition: update deter using previous (deter, stoch) and the action.
        x = torch.cat([deter, stoch, action], dim=-1)
        gru_inp = torch.relu(
            self.inp(x)
        )  # Project concat(deter, stoch, action) then apply ReLU before the GRU update.
        
        deter = self.gru(gru_inp, deter)

        # Prior for the new stochastic latent: p(stoch_t | deter_t)
        stats = self.prior(deter)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        stoch = mean + std * eps  # reparameterized sampling from the diagonal Gaussian prior
        return deter, stoch


class WorldModel(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, cfg: DreamerConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg
        

        # Encoder: e_t = f(o_t)
        self.encoder = mlp(obs_dim, cfg.latent_dim, hidden_dim=cfg.hidden_dim, depth=2)
        self.rssm = RSSM(action_dim, cfg.deter_dim, cfg.latent_dim, hidden_dim=cfg.hidden_dim)

        rssm_feat_dim = cfg.deter_dim + cfg.latent_dim

        # Decoder / observation model: p(o_t | h_t, s_t)
        # Reward model: p(r_t | h_t, s_t)
        self.decoder = mlp(
            rssm_feat_dim,
            (2 * obs_dim if cfg.learn_std else obs_dim),
            hidden_dim=cfg.hidden_dim,
            depth=2,
        )
        self.reward_head = mlp(
            rssm_feat_dim,
            (2 if cfg.learn_std else 1),
            hidden_dim=cfg.hidden_dim,
            depth=2,
        )
        # Continuation / discount model: p(continuation_t | h_t, s_t)
        self.continuation_head = mlp(rssm_feat_dim, 1, hidden_dim=cfg.hidden_dim, depth=2)

    def rssm_features(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        return torch.cat([deter, stoch], dim=-1)

    def _split_mean_log_std(self, stats: torch.Tensor, out_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:

        mean, log_std = torch.split(stats, [out_dim, out_dim], dim=-1)

        log_std = torch.clamp(log_std, float(self.cfg.log_std_min), float(self.cfg.log_std_max))

        return mean, log_std

    def forward_reconstruction(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        obs: (B, T, obs_dim)
        actions: (B, T, action_dim)

        T is the sequence length (configured via `--dreamer-seq-len` in the runner).

        This function runs the world model forward on real sequences for reconstruction-style training:
        - encode obs_t -> emb_t
        - do one RSSM transition (prior rollout)
        - apply posterior correction using emb_t (only available during training on real data)
        - decode latent features to reconstruct obs_t and predict reward r_t
        - compute observation NLL (and diagnostic MSE) and KL(q||p)

        Likelihood details:
        - If cfg.learn_std=True, the decoder and reward head predict both mean and log_std.
          Predicted log_std is clamped to [cfg.log_std_min, cfg.log_std_max].
        - If cfg.learn_std=False, fixed std is used via cfg.obs_std / cfg.reward_std.

        Returns a dict with:
        - features / rssm_features: (B, T, deter_dim + stoch_dim)
        - obs_hat: (B, T, obs_dim)
        - reward_hat: (B, T, 1)
        - reward_log_std: (B, T, 1) (useful for debugging learned uncertainty)
        - obs_log_std_mean: (B, T, 1) mean log_std across obs dims (useful for debugging)
        - obs_nll: scalar (mean over batch and time)
        - obs_mse: scalar (mean over batch and time)
        - kl_loss: scalar (balanced + free-nats; mean over batch and time)
        - kl_raw: scalar (raw KL(q||p); mean over batch and time)
        """

        b, t, _ = obs.shape
        device = obs.device
        deter, stoch = self.rssm.init_state(b, device)

        obs_nll_per_timestep = []  # Per-timestep observation NLL (Gaussian), later averaged into a scalar.

        obs_mse_per_timestep = []  # Per-timestep observation MSE, logged as a metric.

        reward_mean_per_timestep = []  # Per-timestep reward mean predictions stacked to (B, T, 1).

        reward_log_std_per_timestep = []  # Per-timestep reward log_std stacked to (B, T, 1).
        
        obs_mean_per_timestep = []  # Reconstructed obs mean stacked to (B, T, obs_dim).
        obs_log_std_mean_per_timestep = []  # Per-timestep mean log_std over obs dims, stacked to (B, T, 1).
        continuation_logit_per_timestep = []  # Per-timestep continuation logits stacked to (B, T, 1).
        rssm_feature_per_timestep = []  # Per-timestep RSSM features (concat(deter, stoch)).
        kl_loss_per_timestep = []  # Per-timestep (balanced, free-nats) KL, later averaged into a scalar.
        kl_raw_per_timestep = []  # Per-timestep raw KL(q||p), for logging/diagnostics.

        obs_nll_constant = 0.5 * float(torch.log(torch.tensor(2.0 * torch.pi)))

        def diag_gaussian_kl(
            mean_q: torch.Tensor, log_std_q: torch.Tensor, mean_p: torch.Tensor, log_std_p: torch.Tensor
        ) -> torch.Tensor:
            """
            KL(N(mean_q, std_q^2) || N(mean_p, std_p^2)) for diagonal Gaussians.
            Returns shape (B, 1), summed over latent dims.
            """
            var_q = torch.exp(2.0 * log_std_q)
            var_p = torch.exp(2.0 * log_std_p)
            kl_per_dim = (log_std_p - log_std_q) + (var_q + (mean_q - mean_p).pow(2)) / (2.0 * var_p) - 0.5
            return kl_per_dim.sum(dim=-1, keepdim=True)

        for i in range(t):
            emb = self.encoder(obs[:, i])
            deter, stoch = self.rssm.prior_step(deter, stoch, actions[:, i])

            # Prior for the current deter: p(stoch_t | deter_t)
            prior_stats = self.rssm.prior(deter)
            prior_mean, prior_log_std = torch.chunk(prior_stats, 2, dim=-1)
            prior_log_std = torch.clamp(prior_log_std, -20, 2)
            prior_std = torch.exp(prior_log_std)

            # Posterior correction (during training, use encoded observation to correct the latent stochastic state).
            # q(stoch_t | deter_t, emb_t)
            posterior_stats = self.rssm.posterior(torch.cat([deter, emb], dim=-1))
            posterior_mean, posterior_log_std = torch.chunk(posterior_stats, 2, dim=-1)
            posterior_log_std = torch.clamp(posterior_log_std, -5, 2)
            posterior_std = torch.exp(posterior_log_std)

            # KL balancing: distribute gradients between posterior and prior.
            kl_raw = diag_gaussian_kl(posterior_mean, posterior_log_std, prior_mean, prior_log_std)
            kl_raw_per_timestep.append(kl_raw)

            kl_post = diag_gaussian_kl(
                posterior_mean, posterior_log_std, prior_mean.detach(), prior_log_std.detach()
            )  # grads to posterior only
            kl_prior = diag_gaussian_kl(
                posterior_mean.detach(), posterior_log_std.detach(), prior_mean, prior_log_std
            )  # grads to prior only
            kl_bal = float(self.cfg.kl_balance) * kl_post + (1.0 - float(self.cfg.kl_balance)) * kl_prior

            # Free nats: avoid over-regularizing by enforcing a minimum KL.
            free_nats = float(self.cfg.free_nats)
            if free_nats > 0.0:
                kl_bal = torch.clamp(kl_bal, min=free_nats)
            kl_loss_per_timestep.append(kl_bal)
            stoch = posterior_mean + posterior_std * torch.randn_like(posterior_mean)

            rssm_feat = self.rssm_features(deter, stoch)
            rssm_feature_per_timestep.append(rssm_feat)
            if self.cfg.learn_std:
                obs_stats = self.decoder(rssm_feat)
                obs_mean, obs_log_std = self._split_mean_log_std(obs_stats, self.obs_dim)
                reward_stats = self.reward_head(rssm_feat)
                reward_mean, reward_log_std = self._split_mean_log_std(reward_stats, 1)
            else:
                obs_mean = self.decoder(rssm_feat)
                obs_log_std = torch.full_like(
                    obs_mean, float(torch.log(torch.tensor(float(self.cfg.obs_std), device=obs_mean.device)))
                )
                reward_mean = self.reward_head(rssm_feat)
                reward_log_std = torch.full_like(
                    reward_mean, float(torch.log(torch.tensor(float(self.cfg.reward_std), device=reward_mean.device)))
                )
            continuation_logit = self.continuation_head(rssm_feat)
            obs_mean_per_timestep.append(obs_mean)
            reward_mean_per_timestep.append(reward_mean)
            reward_log_std_per_timestep.append(reward_log_std)
            obs_log_std_mean_per_timestep.append(obs_log_std.mean(dim=-1, keepdim=True))
            continuation_logit_per_timestep.append(continuation_logit)

            obs_err = obs_mean - obs[:, i]
            obs_mse_per_timestep.append(obs_err.pow(2).mean(dim=-1, keepdim=True))
            # Gaussian NLL (diagonal). Average over obs dims to match MSE scaling style.
            obs_nll_per_timestep.append(
                (0.5 * (obs_err.pow(2) * torch.exp(-2.0 * obs_log_std)) + obs_log_std + obs_nll_constant)
                .mean(dim=-1, keepdim=True)
            )

        rssm_feats_t = torch.stack(rssm_feature_per_timestep, dim=1)
        obs_hat_t = torch.stack(obs_mean_per_timestep, dim=1)
        r_hat_t = torch.stack(reward_mean_per_timestep, dim=1)
        r_log_std_t = torch.stack(reward_log_std_per_timestep, dim=1)
        obs_log_std_mean_t = torch.stack(obs_log_std_mean_per_timestep, dim=1)
        continuation_logit_t = torch.stack(continuation_logit_per_timestep, dim=1)

        obs_nll = torch.cat(obs_nll_per_timestep, dim=1).mean()
        obs_mse = torch.cat(obs_mse_per_timestep, dim=1).mean()
        kl_loss = torch.cat(kl_loss_per_timestep, dim=1).mean()
        kl_raw = torch.cat(kl_raw_per_timestep, dim=1).mean()

        return {
            "features": rssm_feats_t,  # keep original key for compatibility
            "rssm_features": rssm_feats_t,
            "obs_hat": obs_hat_t,
            "reward_hat": r_hat_t,
            "reward_log_std": r_log_std_t,
            "obs_log_std_mean": obs_log_std_mean_t,
            "continuation_logit": continuation_logit_t,
            "obs_nll": obs_nll,
            "obs_mse": obs_mse,
            "kl_loss": kl_loss,
            "kl_raw": kl_raw,
        }

    @torch.no_grad()
    def infer_posterior_last_state(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Burn-in / state inference for Dreamer-style training.

        This runs the RSSM forward on a real sequence using posterior correction and returns the posterior latent
        state at the final timestep. Conceptually, it maps a real sequence (o_{1:T}, a_{1:T}) to a latent state
        (h_T, s_T), which is then used as the starting state for imagination rollouts.

        Notation mapping (Dreamer-v1 paper):
        - deter ≈ h_t (deterministic GRU memory state)
        - stoch ≈ s_t (stochastic latent sampled from a diagonal Gaussian)

        Args:
            obs: (B, T, obs_dim)
            actions: (B, T, action_dim)

        Returns:
            deter_T: (B, deter_dim)  # deterministic state at t = T
            stoch_T: (B, stoch_dim)  # stochastic state at t = T
        """
        b, t, _ = obs.shape
        device = obs.device
        deter, stoch = self.rssm.init_state(b, device)

        for i in range(t):
            emb = self.encoder(obs[:, i])
            deter, stoch = self.rssm.prior_step(deter, stoch, actions[:, i])
            posterior_stats = self.rssm.posterior(torch.cat([deter, emb], dim=-1))
            posterior_mean, posterior_log_std = torch.chunk(posterior_stats, 2, dim=-1)
            posterior_log_std = torch.clamp(posterior_log_std, -5, 2)
            posterior_std = torch.exp(posterior_log_std)
            stoch = posterior_mean + posterior_std * torch.randn_like(posterior_mean)

        return deter, stoch

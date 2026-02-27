"""
Simplified Dreamer-style agent for low-dimensional.

1. Replace CNN with MLP: Pendulum-v1 has a 3D state, so we drop Dreamer's high-dimensional CNNs and use MLPs for encoder/decoder.
2. RSSM (Recurrent State-Space Model): models stochastic action-conditioned transitions in latent space for multi-step imagination.
3. World model training (paper-shaped, simplified):
   - observation loss: diagonal Gaussian NLL (plus logged MSE for diagnostics)
   - reward loss: diagonal Gaussian NLL (uses `reward_log_std` if `cfg.learn_std=True`)
   - continuation loss: BCE on p(cont_t) vs (1 - done_t)
   - KL regularizer: KL(q||p) with optional balancing + free nats
4. Latent-space actor-critic optimization:
   - imagination rollout uses RSSM prior_step + actor actions
   - lambda-returns use learned continuation p(cont_t) as discount multiplier
   - actor objective optionally includes an entropy bonus controlled by `DreamerConfig.actor_entropy_coef`
5. Policy distribution details:
   - actor samples from a Gaussian, then applies tanh squashing and `action_scale`
   - log-prob uses the exact tanh-squashed Gaussian correction (Jacobian of tanh and scaling), not an approximation

Implementation note (important for correctness):
- Sequence training assumes sampled sequences do not cross episode boundaries; the replay buffer sampler enforces this.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_utils import mlp
from simplified_dreamer_model import DreamerConfig, WorldModel


class Actor(nn.Module):
    def __init__(
        self,
        rssm_feat_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_scale: float = 2.0,
        *,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.net = mlp(rssm_feat_dim, 2 * action_dim, hidden_dim=hidden_dim, depth=2)
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, feat: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tanh-squashed Gaussian action and its log-probability.

        - We sample a pre-squash action u ~ N(mean, std^2).
        - Squash via y = tanh(u), then scale to action space: a = action_scale * y.
        - The returned `logp` is log pi(a | feat) with the exact tanh + scaling correction:
          log pi(a) = log N(u; mean, std) - sum log(action_scale * (1 - tanh(u)^2)).
        """
        stats = self.net(feat)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if deterministic:
            action = torch.tanh(mean) * self.action_scale
            logp = torch.zeros((feat.shape[0], 1), device=feat.device)
            return action, logp
        eps = torch.randn_like(mean)
        u = mean + std * eps
        y = torch.tanh(u)
        a = y * self.action_scale

        # Exact log-prob for tanh-squashed Gaussian with additional action_scale:
        # log p(a) = log p(u) - sum log(action_scale * (1 - tanh(u)^2))
        logp_u = (-0.5 * (((u - mean) / (std + 1e-8)).pow(2) + 2.0 * log_std + torch.log(torch.tensor(2.0 * torch.pi, device=feat.device)))).sum(
            dim=-1, keepdim=True
        )
        log_det = torch.log(torch.as_tensor(self.action_scale, device=feat.device)) * float(self.action_dim) + torch.log(
            1.0 - y.pow(2) + 1e-6
        ).sum(dim=-1, keepdim=True)
        logp = logp_u - log_det
        return a, logp


class Critic(nn.Module):
    def __init__(self, rssm_feat_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = mlp(rssm_feat_dim, 1, hidden_dim=hidden_dim, depth=2)

    def forward(self, rssm_feature: torch.Tensor) -> torch.Tensor:
        return self.net(rssm_feature)


class DreamerAgent:

    def __init__(self, obs_dim: int, action_dim: int, device: Optional[torch.device] = None, cfg: Optional[DreamerConfig] = None):
        self.cfg = cfg or DreamerConfig()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.wm = WorldModel(obs_dim, action_dim, self.cfg).to(self.device)
        rssm_feat_dim = self.cfg.deter_dim + self.cfg.latent_dim
        self.actor = Actor(
            rssm_feat_dim,
            action_dim,
            hidden_dim=self.cfg.hidden_dim,
            log_std_min=float(getattr(self.cfg, "log_std_min", -5.0)),
            log_std_max=float(getattr(self.cfg, "log_std_max", 2.0)),
        ).to(self.device)
        self.critic = Critic(rssm_feat_dim, hidden_dim=self.cfg.hidden_dim).to(self.device)
        self.critic_target = Critic(rssm_feat_dim, hidden_dim=self.cfg.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=self.cfg.model_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        self._last_feat = None
        # Recurrent latent state for acting in the real environment (Dreamer-style).
        self._deter: Optional[torch.Tensor] = None  # (1, deter_dim)
        self._stoch: Optional[torch.Tensor] = None  # (1, latent_dim)
        self._prev_action: Optional[torch.Tensor] = None  # (1, action_dim)

    def reset_episode(self) -> None:
        """
        Reset the recurrent RSSM state used for acting in the real environment.

        This should be called whenever the environment resets (episode boundary), otherwise the latent state would
        incorrectly carry information across episodes.
        """
        with torch.no_grad():
            deter, stoch = self.wm.rssm.init_state(batch=1, device=self.device)
            self._deter = deter
            self._stoch = stoch
            self._prev_action = torch.zeros((1, self.actor.action_dim), device=self.device)

    def select_action(self, obs_np, deterministic: bool = False):
        """
        Select an action in the environment using the current policy.
        Given a single observation `obs_np`, return a single action (numpy).

        Acting procedure:
        - maintain recurrent RSSM state (deter, stoch) across environment steps
        - update deter and sample a prior stoch using the previous action a_{t-1}
        - apply posterior correction using the current observation embedding emb_t
        - feed rssm_features(deter, stoch) to the actor to get the action a_t

        """
        # Convert a single observation (obs_dim,) into a batch of size 1: (1, obs_dim).
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # Ensure the recurrent latent state is initialized before applying prior/posterior updates.
            if self._deter is None or self._stoch is None or self._prev_action is None:
                self.reset_episode()

            emb = self.wm.encoder(obs)

            # 1) prior transition using previous action a_{t-1}
            deter, stoch = self.wm.rssm.prior_step(self._deter, self._stoch, self._prev_action)

            # 2) posterior correction using current observation embedding emb_t
            posterior_stats = self.wm.rssm.posterior(torch.cat([deter, emb], dim=-1))
            posterior_mean, posterior_log_std = torch.chunk(posterior_stats, 2, dim=-1)
            posterior_log_std = torch.clamp(
                posterior_log_std,
                float(getattr(self.cfg, "log_std_min", -5.0)),
                float(getattr(self.cfg, "log_std_max", 2.0)),
            )
            posterior_std = torch.exp(posterior_log_std)
            stoch = posterior_mean + posterior_std * torch.randn_like(posterior_mean)

            # 3) actor input features
            rssm_feat = self.wm.rssm_features(deter, stoch)
            action, _ = self.actor(rssm_feat, deterministic=deterministic)

            # 4) persist state for the next environment step
            self._deter, self._stoch, self._prev_action = deter, stoch, action
        return action.squeeze(0).cpu().numpy()

    def train_world_model(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        done_seq: torch.Tensor,
    ) -> Dict[str, float]:
        self.wm.train()

        out = self.wm.forward_reconstruction(obs_seq, act_seq)
        pred_reward = out["reward_hat"]
        pred_reward_log_std = out["reward_log_std"]
        pred_continuation_logit = out["continuation_logit"]

        # Observation loss (Gaussian NLL) is computed inside the world model forward pass.
        obs_nll = out["obs_nll"]
        obs_mse = out["obs_mse"]

        # Reward loss (Gaussian NLL) computed here because it needs reward targets.
        reward_nll_const = 0.5 * float(torch.log(torch.tensor(2.0 * torch.pi)))
        reward_err = pred_reward - reward_seq
        reward_nll = (
            0.5 * (reward_err.pow(2) * torch.exp(-2.0 * pred_reward_log_std)) + pred_reward_log_std + reward_nll_const
        ).mean()
        reward_mse = reward_err.pow(2).mean()

        # Continuation loss: continuation_target = 1 for non-terminal, 0 for terminal.
        continuation_target = (1.0 - done_seq).clamp(0.0, 1.0)
        continuation_loss = F.binary_cross_entropy_with_logits(pred_continuation_logit, continuation_target)

        kl_loss = out["kl_loss"]
        kl_raw = out["kl_raw"]

        # Total world-model loss (Dreamer-style core pieces):
        # - obs_nll: observation reconstruction likelihood
        # - reward_nll: reward prediction likelihood
        # - continuation_beta * continuation_loss: continuation/discount modeling
        # - kl_beta * kl_loss: regularize posterior against prior
        total_loss = (
            obs_nll + reward_nll + self.cfg.continuation_beta * continuation_loss + self.cfg.kl_beta * kl_loss
        )

        self.wm_opt.zero_grad()
        total_loss.backward()
        self.wm_opt.step()
        return {
            "wm/total_loss": float(total_loss.item()),
            # Backward-compatible alias (older dashboards may still look for wm/loss).
            "wm/loss": float(total_loss.item()),
            "wm/obs_nll": float(obs_nll.item()),
            "wm/reward_nll": float(reward_nll.item()),
            # Backward-compatible alias.
            "wm/cont_bce": float(continuation_loss.item()),
            "wm/continuation_bce": float(continuation_loss.item()),
            "wm/obs_mse": float(obs_mse.item()),
            "wm/reward_mse": float(reward_mse.item()),
            "wm/reward_avg_log_std": float(pred_reward_log_std.mean().item()),
            "wm/obs_avg_log_std": float(out["obs_log_std_mean"].mean().item()),
            "wm/kl": float(kl_loss.item()),
            "wm/kl_raw": float(kl_raw.item()),
        }

    def train_actor_critic(
        self, obs_seq: torch.Tensor, action_seq: torch.Tensor
    ) -> Dict[str, float]:
        """
        Dreamer-style actor/value training via imagination rollouts (simplified but paper-shaped).

        High-level:
        1) Infer a posterior RSSM state from a real sequence (burn-in / state inference).
        2) Imagine forward for H steps using the actor and RSSM prior.
        3) Compute lambda-returns from predicted rewards and predicted values.
        4) Update value network to fit lambda-returns; update actor to maximize lambda-returns.

        Actor regularization:
        - If `cfg.actor_entropy_coef > 0`, the actor objective becomes:
          maximize (lambda_returns + entropy_coef * entropy)
          which we implement as minimizing: -(lambda_returns - entropy_coef * logpi(a)).
        """
        self.actor.train()
        self.critic.train()

        # 1) Infer posterior state at the end of the real sequence (burn-in).
        deter, stoch = self.wm.infer_posterior_last_state(obs_seq, action_seq)  # (B, deter_dim), (B, stoch_dim)

        horizon = int(self.cfg.horizon)
        gamma = float(self.cfg.gamma)
        lambda_ = float(getattr(self.cfg, "lambda_", 0.95))

        # 2) Imagine forward for H steps using the RSSM prior and actor actions.
        imagined_rewards = []
        imagined_continuation_probs = []
        imagined_features = []
        imagined_action_logps = []

        # Helper: temporarily freeze/unfreeze a module's parameters.
        #
        # In the imagination rollout, the actor loss depends on computations that go through the world model and the
        # value network. During the *actor* update step we want gradients to flow through these computations with
        # respect to the actor inputs (actions / latent features), but we do NOT want to update the parameters of the
        # world model or the value network at the same time. Freezing their parameters keeps them fixed while still
        # allowing the computation graph to be differentiable w.r.t. the actor.
        def _set_requires_grad(module: nn.Module, flag: bool) -> None:
            for p in module.parameters():
                p.requires_grad_(flag)

        # Feature at z_0 (start state)
        rssm_feature = self.wm.rssm_features(deter, stoch)
        imagined_features.append(rssm_feature)

        for _ in range(horizon):
            rssm_feature = self.wm.rssm_features(deter, stoch)
            action, action_logp = self.actor(rssm_feature, deterministic=False)
            deter, stoch = self.wm.rssm.prior_step(deter, stoch, action)
            next_feature = self.wm.rssm_features(deter, stoch)
            reward_out = self.wm.reward_head(next_feature)
            if bool(getattr(self.cfg, "learn_std", True)):
                reward_pred, _ = torch.chunk(reward_out, 2, dim=-1)
            else:
                reward_pred = reward_out
            continuation_prob = torch.sigmoid(self.wm.continuation_head(next_feature))
            imagined_rewards.append(reward_pred)
            imagined_continuation_probs.append(continuation_prob)
            imagined_features.append(next_feature)
            imagined_action_logps.append(action_logp)

        reward_seq_imag = torch.stack(imagined_rewards, dim=1)  # (B, H, 1)
        continuation_prob_seq_imag = torch.stack(imagined_continuation_probs, dim=1)  # (B, H, 1)
        feature_seq_imag = torch.stack(imagined_features, dim=1)  # (B, H+1, D)
        action_logp_seq_imag = torch.stack(imagined_action_logps, dim=1)  # (B, H, 1)

        # 3) Actor update: maximize lambda-returns through imagination.
        # Freeze world model and value network parameters (treat them as constants) but keep the computation
        # differentiable w.r.t. the actor through the imagined trajectory.
        _set_requires_grad(self.wm, False)
        _set_requires_grad(self.critic_target, False)
        try:
            # Compute V(z_t) along the imagined trajectory (parameters frozen, gradients flow to inputs).
            b, hp1, d = feature_seq_imag.shape
            value_seq_imag = self.critic_target(feature_seq_imag.reshape(b * hp1, d)).reshape(b, hp1, 1)  # (B, H+1, 1)

            actor_returns = torch.zeros_like(reward_seq_imag)  # (B, H, 1)

            G = value_seq_imag[:, -1]  # bootstrap with V(z_H)

            # Compute lambda-returns from predicted rewards and predicted values.
            for i in reversed(range(horizon)):
                v_next = value_seq_imag[:, i + 1]
                discount = gamma * continuation_prob_seq_imag[:, i]
                G = reward_seq_imag[:, i] + discount * ((1.0 - lambda_) * v_next + lambda_ * G)
                actor_returns[:, i] = G

            ent_coef = float(getattr(self.cfg, "actor_entropy_coef", 0.0))
            actor_loss = -(actor_returns - ent_coef * action_logp_seq_imag).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        finally:
            _set_requires_grad(self.wm, True)
            _set_requires_grad(self.critic_target, True)

        # 4) Value update: fit V(z_t) to lambda-return targets (stop-gradient targets).
        # - Build a fixed value sequence V(z_0..z_H) from the imagined features (no gradients).
        # - Use it to compute fixed lambda-return targets G^λ_0..G^λ_{H-1}.
        # - Regress the value network V(z_t) onto these targets.
        with torch.no_grad():
            # Detach imagined features so the value-loss gradients do not flow back into the imagination trajectory.
            feature_seq_detached = feature_seq_imag.detach()
            b, hp1, d = feature_seq_detached.shape
            value_seq_target = self.critic_target(feature_seq_detached.reshape(b * hp1, d)).reshape(b, hp1, 1).detach()

            target_returns = torch.zeros_like(reward_seq_imag)
            G = value_seq_target[:, -1]
            for i in reversed(range(horizon)):
                v_next = value_seq_target[:, i + 1]
                discount = gamma * continuation_prob_seq_imag[:, i].detach()
                G = reward_seq_imag[:, i].detach() + discount * ((1.0 - lambda_) * v_next + lambda_ * G)
                target_returns[:, i] = G

        b, hp1, d = feature_seq_imag.shape
        value_pred_seq = self.critic(feature_seq_imag.detach()[:, :-1].reshape(b * (hp1 - 1), d)).reshape(b, hp1 - 1, 1)
        value_loss = F.mse_loss(value_pred_seq, target_returns)
        self.critic_opt.zero_grad()
        value_loss.backward()
        self.critic_opt.step()

        # Soft update target value network for stability.
        tau = float(getattr(self.cfg, "target_tau", 0.01))
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

        return {
            "loss/value": float(value_loss.item()),
            "loss/actor": float(actor_loss.item()),
            "actor/entropy_est": float((-action_logp_seq_imag.detach()).mean().item()),
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "cfg": {
                "latent_dim": self.cfg.latent_dim,
                "deter_dim": self.cfg.deter_dim,
                "hidden_dim": self.cfg.hidden_dim,
                "horizon": self.cfg.horizon,
                "model_lr": self.cfg.model_lr,
                "actor_lr": self.cfg.actor_lr,
                "critic_lr": self.cfg.critic_lr,
                "kl_beta": self.cfg.kl_beta,
                "continuation_beta": self.cfg.continuation_beta,
                "free_nats": self.cfg.free_nats,
                "kl_balance": self.cfg.kl_balance,
                "learn_std": bool(getattr(self.cfg, "learn_std", True)),
                "obs_std": self.cfg.obs_std,
                "reward_std": self.cfg.reward_std,
                "log_std_min": float(getattr(self.cfg, "log_std_min", -5.0)),
                "log_std_max": float(getattr(self.cfg, "log_std_max", 2.0)),
                "target_tau": self.cfg.target_tau,
                "lambda_": float(getattr(self.cfg, "lambda_", 0.95)),
                "actor_entropy_coef": float(getattr(self.cfg, "actor_entropy_coef", 0.0)),
                "gamma": self.cfg.gamma,
            },
            "wm": self.wm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "wm_opt": self.wm_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.wm.load_state_dict(state["wm"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        if "critic_target" in state:
            self.critic_target.load_state_dict(state["critic_target"])
        else:
            self.critic_target.load_state_dict(state["critic"])
        self.wm_opt.load_state_dict(state["wm_opt"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])


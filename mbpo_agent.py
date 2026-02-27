"""
MBPO agent

Core idea of MBPO: train a dynamics model on real environment transitions, then use the model to generate short
"imagined" rollouts (synthetic transitions) and feed them to SAC for additional off-policy updates.

- Policy learning: reuse SAC (actor/critic) but allow training from explicit tensors.
- Model learning: learn an ensemble dynamics model from real transitions.
- Data augmentation: perform synthetic rollouts from model to generate additional transitions,
  then train SAC on a mixture of real + synthetic batches.

Overall order (per env step) in the runner:
- Interact with the real environment:
  - action = agent.select_action(...) (random actions during prefill)
  - next_state, reward = env.step(action)
  - replay_buffer.add(state, action, reward, next_state, done)  # real data goes into the real buffer
- If step >= replay_prefill_steps, perform updates:
  2.1 Real SAC update (from the real replay buffer)
      - agent.train_policy_on_real(replay_buffer)
  2.2 Model update (train dynamics ensemble from the real replay buffer)
      - agent.train_model(replay_buffer)
      - this updates ensemble.selected_model_indices (top-k selection)
  2.3 Synthetic SAC updates (rollout happens here)
      - agent.train_policy_on_synthetic(replay_buffer)
      - sample start states from the real buffer: s0 = state_batch
      - rollout_model(s0, horizon=H) generates synthetic (s, a, r, s', done)
      - immediately run SAC updates on the synthetic batch via train_from_tensors(...)
      - note: synthetic transitions are NOT written back to replay_buffer in this minimal implementation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from sac_agent import SACAgent
from mbpo_models import DynamicsEnsemble, ModelTrainStats, train_dynamics_ensemble


@dataclass
class MBPOConfig:
    horizon: int = 5
    model_ensemble_size: int = 7
    model_top_k: int = 5
    model_hidden_dim: int = 256
    model_lr: float = 1e-3
    model_train_steps_per_env_step: int = 1
    synthetic_updates_per_env_step: int = 1


class MBPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: Optional[torch.device] = None,
        sac_kwargs: Optional[Dict[str, Any]] = None,
        mbpo_cfg: Optional[MBPOConfig] = None,
    ):
        sac_kwargs = sac_kwargs or {}
        self.device = device
        self.policy = SACAgent(state_dim, action_dim, device=device, **sac_kwargs)

        self.cfg = mbpo_cfg or MBPOConfig()
        self.model = DynamicsEnsemble(
            state_dim=state_dim,
            action_dim=action_dim,
            ensemble_size=self.cfg.model_ensemble_size,
            hidden_dim=self.cfg.model_hidden_dim,
            top_k_models=self.cfg.model_top_k,
        ).to(self.policy.device)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.model_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state, deterministic: bool = False):
        return self.policy.select_action(state, deterministic=deterministic)

    def train_policy_on_real(self, replay_buffer, batch_size: int) -> Dict[str, float]:
        """
        Perform SAC updates using real environment transitions stored in the replay buffer.

        Returns a Dict[str, float] of scalar losses/metrics (e.g., actor/critic losses) for logging.
        """
        return self.policy.train(replay_buffer, batch_size=batch_size)

    def train_model(self, replay_buffer, batch_size: int) -> ModelTrainStats:
        batch = replay_buffer.sample(batch_size)
        batch = tuple(t.to(self.policy.device) for t in batch)  # type: ignore[assignment]
        stats = None
        for _ in range(max(1, int(self.cfg.model_train_steps_per_env_step))):
            # Key knob: how many gradient steps to train the world model per real env step.
            # Example: model_train_steps_per_env_step=5 means we train the model 5 times per env step
            # (in this minimal implementation, on the same sampled batch).
            stats = train_dynamics_ensemble(self.model, batch, self.model_optimizer)
        # The loop runs at least once due to max(1, ...); stats should never be None here.
        assert stats is not None
        return stats

    @torch.no_grad()
    def rollout_model(
        self,
        start_states: torch.Tensor,
        horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate synthetic transitions using the learned model and current policy.
        Returns tensors shaped like a replay batch: (s, a, r, s', done).
        """
        device = self.policy.device
        s = start_states.to(device)

        # These 5 lists collect per-step rollout tuples (s, a, r, s', done).
        # append one step per loop iteration, then concatenate into a single large batch with
        # shapes roughly (batch_size * horizon, ...), which is fed into SACAgent.train_from_tensors(...)
        # for synthetic updates.
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        for _ in range(horizon):
            # Vectorized policy action via actor (stochastic during training).
            st = s
            action_t, _ = self.policy.actor(st, deterministic=False)  # type: ignore[attr-defined]
            a_t = action_t

            ns, r = self.model.predict(st, a_t)
            done = torch.zeros((st.shape[0], 1), device=device, dtype=st.dtype)

            batch_states.append(st)
            batch_actions.append(a_t)
            batch_rewards.append(r)
            batch_next_states.append(ns)
            batch_dones.append(done)

            s = ns

        return (
            torch.cat(batch_states, dim=0),
            torch.cat(batch_actions, dim=0),
            torch.cat(batch_rewards, dim=0),
            torch.cat(batch_next_states, dim=0),
            torch.cat(batch_dones, dim=0),
        )

    def train_policy_on_synthetic(self, replay_buffer, batch_size: int) -> Dict[str, float]:
        """
        Sample start states from the real buffer, roll out with the model for H steps,
        then train SAC on the generated synthetic batch.
        """
        state, _, _, _, _ = replay_buffer.sample(batch_size)
        s0 = state.to(self.policy.device)

        losses: Dict[str, float] = {}
        for _ in range(max(1, int(self.cfg.synthetic_updates_per_env_step))):
            syn_batch = self.rollout_model(s0, horizon=self.cfg.horizon)
            last = self.policy.train_from_tensors(*syn_batch)
            losses = last
        return losses

    def get_state(self) -> Dict[str, Any]:
        return {
            "mbpo_cfg": {
                "horizon": self.cfg.horizon,
                "model_ensemble_size": self.cfg.model_ensemble_size,
                "model_hidden_dim": self.cfg.model_hidden_dim,
                "model_lr": self.cfg.model_lr,
                "model_train_steps_per_env_step": self.cfg.model_train_steps_per_env_step,
                "synthetic_updates_per_env_step": self.cfg.synthetic_updates_per_env_step,
            },
            "policy": self.policy.get_state(),
            "model": self.model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.policy.load_state(state["policy"])
        self.model.load_state_dict(state["model"])
        self.model_optimizer.load_state_dict(state["model_optimizer"])


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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from sac_agent import SACAgent, SACConfig
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
    # Which replay signal trains the terminal head / synthetic done:
    # - "terminated": done_for_learning (physics terminal only)
    # - "episode_end": terminated OR truncated (finite-horizon / time-limit as terminal)
    terminal_target: str = "terminated"


class MBPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: Optional[torch.device] = None,
        sac_cfg: Optional[SACConfig] = None,
        sac_kwargs: Optional[Dict[str, Any]] = None,
        mbpo_cfg: Optional[MBPOConfig] = None,
    ):
        self.device = device
        # MBPO uses SAC as the policy learner.
        #
        # Preferred: pass `sac_cfg=SACConfig(...)` (unified config style).
        # Backward-compatible: accept sac_kwargs and convert to SACConfig.
        if sac_cfg is None:
            sac_kwargs = sac_kwargs or {}
            sac_cfg_kwargs = {k: v for k, v in sac_kwargs.items() if k in SACConfig.__dataclass_fields__}
            sac_cfg = SACConfig(**sac_cfg_kwargs)
        self.policy = SACAgent(state_dim, action_dim, device=device, cfg=sac_cfg)

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
        n_steps = max(1, int(self.cfg.model_train_steps_per_env_step))
        stats_last = None
        acc = {
            "loss": 0.0,
            "nll": 0.0,
            "mse_next_state": 0.0,
            "mse_reward": 0.0,
            "avg_log_std": 0.0,
            "epistemic_var_next_state": 0.0,
            "epistemic_var_reward": 0.0,
            "terminated_bce": 0.0,
            "terminated_acc": 0.0,
            "terminated_rate": 0.0,
        }
        for _ in range(n_steps):
            # Key knob: how many gradient steps to train the world model per real env step.
            # Example: model_train_steps_per_env_step=5 means we train the model 5 times per env step
            # (each step re-samples a fresh replay batch for stability).
            batch6 = replay_buffer.sample_with_episode_end(batch_size)
            batch6 = tuple(t.to(self.policy.device) for t in batch6)  # type: ignore[assignment]
            state, action, reward, next_state, done_for_learning, episode_end = batch6
            term_mode = str(getattr(self.cfg, "terminal_target", "terminated"))
            done_target = episode_end if term_mode == "episode_end" else done_for_learning
            batch5 = (state, action, reward, next_state, done_target)
            stats_last = train_dynamics_ensemble(self.model, batch5, self.model_optimizer)
            acc["loss"] += float(stats_last.loss)
            acc["nll"] += float(stats_last.nll)
            acc["mse_next_state"] += float(stats_last.mse_next_state)
            acc["mse_reward"] += float(stats_last.mse_reward)
            acc["avg_log_std"] += float(stats_last.avg_log_std)
            acc["epistemic_var_next_state"] += float(stats_last.epistemic_var_next_state)
            acc["epistemic_var_reward"] += float(stats_last.epistemic_var_reward)
            acc["terminated_bce"] += float(getattr(stats_last, "terminated_bce", 0.0))
            acc["terminated_acc"] += float(getattr(stats_last, "terminated_acc", 0.0))
            acc["terminated_rate"] += float(getattr(stats_last, "terminated_rate", 0.0))

        assert stats_last is not None
        inv = 1.0 / float(n_steps)
        return ModelTrainStats(
            nll=acc["nll"] * inv,
            loss=acc["loss"] * inv,
            mse_next_state=acc["mse_next_state"] * inv,
            mse_reward=acc["mse_reward"] * inv,
            avg_log_std=acc["avg_log_std"] * inv,
            epistemic_var_next_state=acc["epistemic_var_next_state"] * inv,
            epistemic_var_reward=acc["epistemic_var_reward"] * inv,
            selected_model_indices=list(stats_last.selected_model_indices),
            terminated_bce=acc["terminated_bce"] * inv,
            terminated_acc=acc["terminated_acc"] * inv,
            terminated_rate=acc["terminated_rate"] * inv,
        )

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
        done_so_far = torch.zeros((s.shape[0], 1), device=device, dtype=s.dtype)

        for _ in range(horizon):
            # Vectorized policy action via actor (stochastic during training).
            st = s
            action_t, _ = self.policy.actor(st, deterministic=False)  # type: ignore[attr-defined]
            # For already-terminated synthetic states, use a dummy action (absorbing state convention).
            already_done = done_so_far > 0.5
            a_t = torch.where(already_done.expand_as(action_t), torch.zeros_like(action_t), action_t)

            ns_pred, r_pred, done_pred = self.model.predict(st, a_t)
            done_pred = (done_pred > 0.5).to(dtype=st.dtype)
            done_t = torch.clamp(done_so_far + done_pred, 0.0, 1.0)

            # Absorbing / terminal handling for synthetic rollouts:
            # once done=1, keep state fixed and reward=0 for remaining rollout steps.
            ns = torch.where(already_done, st, ns_pred)
            r = torch.where(already_done, torch.zeros_like(r_pred), r_pred)

            batch_states.append(st)
            batch_actions.append(a_t)
            batch_rewards.append(r)
            batch_next_states.append(ns)
            batch_dones.append(done_t)

            s = ns
            done_so_far = done_t

        return (
            torch.cat(batch_states, dim=0),
            torch.cat(batch_actions, dim=0),
            torch.cat(batch_rewards, dim=0),
            torch.cat(batch_next_states, dim=0),
            torch.cat(batch_dones, dim=0),
        )

    def train_policy_on_synthetic(self, replay_buffer, batch_size: int) -> Dict[str, float]:
        """
        Sample start states from the real replay buffer, roll out with the learned model for H steps,
        then train SAC on the generated synthetic batch.

        Notes:
        - If synthetic_updates_per_env_step > 1, we re-sample fresh start states for each synthetic update to improve
          diversity (rather than reusing a single s0 batch).
        - Synthetic `done` is produced by the dynamics ensemble's terminal head (trained on replay done_for_learning),
          and an absorbing-state mask is applied after termination.
        """
        n_updates = max(1, int(self.cfg.synthetic_updates_per_env_step))
        acc: Dict[str, float] = {}
        for _ in range(n_updates):
            state, _, _, _, _ = replay_buffer.sample(batch_size)
            s0 = state.to(self.policy.device)
            syn_batch = self.rollout_model(s0, horizon=self.cfg.horizon)
            last = self.policy.train_from_tensors(*syn_batch)
            for k, v in last.items():
                acc[k] = float(acc.get(k, 0.0) + float(v))
        return {k: v / float(n_updates) for k, v in acc.items()}

    def get_state(self) -> Dict[str, Any]:
        return {
            "mbpo_cfg": {
                "horizon": self.cfg.horizon,
                "model_ensemble_size": self.cfg.model_ensemble_size,
                "model_hidden_dim": self.cfg.model_hidden_dim,
                "model_lr": self.cfg.model_lr,
                "model_train_steps_per_env_step": self.cfg.model_train_steps_per_env_step,
                "synthetic_updates_per_env_step": self.cfg.synthetic_updates_per_env_step,
                "terminal_target": str(getattr(self.cfg, "terminal_target", "terminated")),
            },
            "policy": self.policy.get_state(),
            "model": self.model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.policy.load_state(state["policy"])
        self.model.load_state_dict(state["model"])
        self.model_optimizer.load_state_dict(state["model_optimizer"])


"""
MBPO agent

Core idea of MBPO: train a dynamics model on real environment transitions, then use the model to generate short
"imagined" rollouts (synthetic transitions) and feed them to SAC for additional off-policy updates.

- Policy learning: reuse SAC (actor/critic) but allow training from explicit tensors.
- Model learning: learn an ensemble dynamics model from real transitions.
- Data augmentation: perform synthetic rollouts from model to generate additional transitions, then train SAC on a mixture of real + synthetic batches.

Top k parameters:
5 from 7 models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from buffer import ReplayBuffer
from sac_agent import SACAgent, SACConfig
from mbpo_models import DynamicsEnsemble, ModelTrainStats, train_dynamics_ensemble


@dataclass
class MBPOConfig:
    horizon: int = 5
    model_ensemble_size: int = 7
    model_top_k: int = 5
    model_hidden_dim: int = 256
    model_lr: float = 1e-3
    model_weight_decay: float = 1e-4
    model_normalize: bool = True

    # ================================
    # Model training.
    # ================================
    # For LunarLanderContinuous-v3, we need more aggressive model training and synthetic updates
    model_train_steps_per_env_step: int = 1  # reduced from 5 to speed up Wall-clock time
    synthetic_updates_per_env_step: int = 1 # reduced from 20 to balance speed and sample efficiency
    # Which replay signal trains the terminal head / synthetic done:
    # - "terminated": done_for_learning (physics terminal only)
    # - "episode_end": terminated OR truncated (finite-horizon / time-limit as terminal)
    terminal_target: str = "terminated"


    # ================================
    # Synthetic data usage.
    # ================================
    # Start-state sampling for model rollouts:
    # sample s0 from the most recent fraction of replay (MBPO-style).
    # 0.25 is good for simple envs, but for LunarLander we want to explore more diverse states
    start_state_recent_frac: float = 1.0  # changed from 0.25 to sample uniformly from all history

    # Whether to keep post-termination absorbing transitions in synthetic batch.
    # False (default): drop post-done absorbing tuples to improve effective sample usage.
    keep_absorbing_post_done: bool = False

    # Whether to store synthetic transitions in a dedicated replay buffer and
    # train SAC from that buffer (MBPO-style, improves synthetic data reuse).
    use_synthetic_buffer: bool = True
    synthetic_buffer_size: int = 200000 
    synthetic_min_size: int = 256




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

        # MBPO uses SAC as the policy learner.
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
            normalize_inputs_targets=self.cfg.model_normalize,
        ).to(self.policy.device)

        self.model_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.model_lr,
            weight_decay=float(getattr(self.cfg, "model_weight_decay", 0.0)),
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.synthetic_buffer: Optional[ReplayBuffer] = None
        if self.cfg.use_synthetic_buffer:
            self.synthetic_buffer = ReplayBuffer(
                state_dim=state_dim,
                action_dim=action_dim,
                max_size=max(1, self.cfg.synthetic_buffer_size),
                device=self.policy.device,
            )

    def select_action(self, state, deterministic: bool = False):
        return self.policy.select_action(state, deterministic=deterministic)

    def train_policy_on_real(self, replay_buffer, batch_size: int):
        """
        Perform SAC updates using real environment transitions stored in the replay buffer.

        Returns a Dict[str, float] of scalar losses/metrics (e.g., actor/critic losses) for logging.
        """
        return self.policy.train(replay_buffer, batch_size=batch_size)

    def train_model(self, replay_buffer, batch_size: int) -> ModelTrainStats:
        
        # Defensive programming: ensure n_steps is at least 1.
        n_steps = max(1, int(self.cfg.model_train_steps_per_env_step))

        # Record the last stats.
        stats_last = None

        # Initialize the accumulators for the metrics.
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

            # Sample a batch of transitions with episode end.
            # batch6 is a tuple of 6 tensors: (state, action, reward, next_state, done_for_learning, episode_end).
            batch6 = replay_buffer.sample_with_episode_end(batch_size)
            # Move the tensors to the device.
            batch6 = tuple(t.to(self.policy.device) for t in batch6)  # type: ignore[assignment]
            # Unpack the batch.
            state, action, reward, next_state, done_for_learning, episode_end = batch6
            # Determine the terminal target.
            term_mode = self.cfg.terminal_target
            # Determine the done target.
            done_target = episode_end if term_mode == "episode_end" else done_for_learning
            # Pack the batch into a tuple of 5 tensors: (state, action, reward, next_state, done_target).   
            batch5 = (state, action, reward, next_state, done_target)
            # Train the dynamics ensemble.
            stats_last = train_dynamics_ensemble(self.model, batch5, self.model_optimizer)
            # Accumulate the metrics.
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

        # Return the average metrics.
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
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        done_so_far = torch.zeros((s.shape[0], 1), device=device, dtype=s.dtype)

        # ================================
        # Roll out the model for the given horizon.
        # ================================
        for _ in range(horizon):
            # current state
            st = s
            # sample an action from the policy
            action_t, _ = self.policy.actor(st, deterministic=False) 
            # check if the state is already done
            already_done = done_so_far > 0.5
            a_t = torch.where(already_done.expand_as(action_t), torch.zeros_like(action_t), action_t)

            # *predict the next state and reward
            # ns_pred: next state prediction
            # r_pred: reward prediction
            # done_pred: done prediction
            ns_pred, r_pred, done_pred = self.model.predict(st, a_t)
            done_pred = (done_pred > 0.5).to(dtype=st.dtype)
            done_t = torch.clamp(done_so_far + done_pred, 0.0, 1.0)

            ns = torch.where(already_done, st, ns_pred)
            r = torch.where(already_done, torch.zeros_like(r_pred), r_pred)

            if self.cfg.keep_absorbing_post_done:
                batch_states.append(st)
                batch_actions.append(a_t)
                batch_rewards.append(r)
                batch_next_states.append(ns)
                batch_dones.append(done_t)
            else:
                alive_mask = (~already_done).squeeze(-1)
                if torch.any(alive_mask):
                    batch_states.append(st[alive_mask])
                    batch_actions.append(a_t[alive_mask])
                    batch_rewards.append(r[alive_mask])
                    batch_next_states.append(ns[alive_mask])
                    batch_dones.append(done_t[alive_mask])

            # pass the next state to the next iteration
            # s = ns
            # pass the done to the next iteration
            done_so_far = done_t

        # Concatenate the lists into tensors.
        return (
            torch.cat(batch_states, dim=0),
            torch.cat(batch_actions, dim=0),
            torch.cat(batch_rewards, dim=0),
            torch.cat(batch_next_states, dim=0),
            torch.cat(batch_dones, dim=0),
        )

    def train_policy_on_synthetic(self, replay_buffer, batch_size: int) -> Dict[str, float]:
        """
        * The most important feature of MBPO.
        Train the policy on the synthetic batch.
        The policy is from SACAgent(train_from_tensors).
        """

        n_updates = max(1, int(self.cfg.synthetic_updates_per_env_step))

        acc: Dict[str, float] = {}

        n_effective_updates = 0

        # ================================
        # Generate synthetic data (Rollout) ONCE per env step
        # ================================
        recent_frac = self.cfg.start_state_recent_frac
        if hasattr(replay_buffer, "sample_recent_states"):
            s0 = replay_buffer.sample_recent_states(batch_size, recent_frac=recent_frac).to(self.policy.device)
        else:
            # Backward compatibility with older replay buffers.
            state, _, _, _, _ = replay_buffer.sample(batch_size)
            s0 = state.to(self.policy.device)

        # Roll out the model for the given horizon.
        syn_batch = self.rollout_model(s0, horizon=self.cfg.horizon)  # （s, a, r, s', done）

        # If the synthetic buffer is used, add the synthetic batch to the synthetic buffer.
        if self.cfg.use_synthetic_buffer and self.synthetic_buffer is not None:
            # unpack the synthetic batch
            s, a, r, ns, d = syn_batch

            # convert the synthetic batch to numpy
            s_np = s.detach().cpu().numpy()
            a_np = a.detach().cpu().numpy()
            r_np = r.detach().cpu().numpy()
            ns_np = ns.detach().cpu().numpy()
            d_np = d.detach().cpu().numpy()

            # add the synthetic batch to the synthetic buffer
            for i in range(s_np.shape[0]):
                self.synthetic_buffer.add(
                    s_np[i], a_np[i], r_np[i], ns_np[i], d_np[i], episode_end=float(d_np[i, 0])
                )

        # ================================
        # Train the policy on the synthetic batch.
        # ================================
        for _ in range(n_updates):
            if self.cfg.use_synthetic_buffer and self.synthetic_buffer is not None:
                # if the synthetic buffer is not full, continue
                if self.synthetic_buffer.size < max(1, self.cfg.synthetic_min_size):
                    continue

                # sample a batch from the synthetic buffer
                syn_train_batch = self.synthetic_buffer.sample(batch_size)

                # train the policy on the synthetic batch
                last = self.policy.train_from_tensors(*syn_train_batch)
            else:
                # If not using a buffer, just train on the generated batch directly
                # (Note: this means all n_updates will use the exact same batch, which might overfit)
                last = self.policy.train_from_tensors(*syn_batch)

            n_effective_updates += 1

            for k, v in last.items():
                acc[k] = float(acc.get(k, 0.0) + float(v))

        if n_effective_updates == 0:
            return {}

        return {k: v / float(n_effective_updates) for k, v in acc.items()}

    def get_state(self) -> Dict[str, Any]:
        return {
            "mbpo_cfg": {
                "horizon": self.cfg.horizon,
                "model_ensemble_size": self.cfg.model_ensemble_size,
                "model_top_k": self.cfg.model_top_k,
                "model_hidden_dim": self.cfg.model_hidden_dim,
                "model_lr": self.cfg.model_lr,
                "model_weight_decay": float(getattr(self.cfg, "model_weight_decay", 0.0)),
                "model_normalize": bool(getattr(self.cfg, "model_normalize", True)),
                "model_train_steps_per_env_step": self.cfg.model_train_steps_per_env_step,
                "synthetic_updates_per_env_step": self.cfg.synthetic_updates_per_env_step,
                "start_state_recent_frac": float(getattr(self.cfg, "start_state_recent_frac", 0.25)),
                "keep_absorbing_post_done": bool(getattr(self.cfg, "keep_absorbing_post_done", False)),
                "use_synthetic_buffer": bool(getattr(self.cfg, "use_synthetic_buffer", True)),
                "synthetic_buffer_size": int(getattr(self.cfg, "synthetic_buffer_size", 200000)),
                "synthetic_min_size": int(getattr(self.cfg, "synthetic_min_size", 256)),
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


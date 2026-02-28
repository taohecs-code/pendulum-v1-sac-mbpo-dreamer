"""
Unified experiment runner for Pendulum-v1:
- SAC (model-free baseline)
- MBPO (model-based policy optimization with synthetic rollouts)
- Dreamer (simplified Dreamer-v1 style latent imagination; MLP-only for low-dim Pendulum)

Design goals:
- Same evaluation protocol across algorithms
- Same logging schema (W&B-friendly)
- Same checkpoint naming rules (step / converged / final)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Allow running this script directly via `python experiment/run_pendulum.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import torch

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from sac_agent import SACAgent, SACConfig
from buffer import ReplayBuffer
from mbpo_agent import MBPOAgent, MBPOConfig
from simplified_dreamer_agent import DreamerAgent, DreamerConfig


# =========================
# Constants / defaults
# =========================

# same from experiment/sac_exp_pendulum.py
ENV_NAME_DEFAULT = "Pendulum-v1"
EPISODE_MAX_STEPS_DEFAULT = 200

TRAIN_MAX_STEPS_DEFAULT = 100_000
EVAL_FREQUENCY_STEPS_DEFAULT = 2_000
REPLAY_PREFILL_STEPS_DEFAULT = 1_000
BATCH_SIZE_DEFAULT = 256

CONVERGENCE_THRESHOLD_DEFAULT = -150.0
CONVERGENCE_STABLE_K_DEFAULT = 3
ASYMPTOTIC_WINDOW_EVALS_DEFAULT = 10

EVAL_EPISODES_DURING_TRAIN_DEFAULT = 5
EVAL_EPISODES_FINAL_DEFAULT = 10

# todo
CHECKPOINT_EVERY_STEPS_DEFAULT = 10_000  # step_10k.pt, step_20k.pt, ...

DREAMER_SEQ_LEN_DEFAULT = 50


def pick_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_step_k(step: int) -> str:
    if step % 1000 == 0:
        return f"{step//1000}k"
    return str(step)


def format_eta(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def make_run_name(algo: str, seed: int, horizon: Optional[int] = None, arch: Optional[str] = None) -> str:
    algo = algo.upper()
    parts = [algo]
    if algo == "SAC":
        parts.append("Baseline")
    if horizon is not None and algo in {"MBPO", "DREAMER"}:
        parts.append(f"Horizon{horizon}")
    if arch is not None and algo == "DREAMER":
        parts.append(arch)
    parts.append(f"Seed{seed}")
    return "_".join(parts)


def evaluate_policy(
    # deterministic=True
    agent: Any,
    env_name: str,
    eval_episodes: int,
    seed: int,
    episode_max_steps: int,
) -> float:
    env = gym.make(env_name)
    avg_reward = 0.0
    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        # Some agents (Dreamer-style) maintain recurrent state across steps and must be reset at episode boundaries.
        if hasattr(agent, "reset_episode") and callable(getattr(agent, "reset_episode")):
            agent.reset_episode()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < episode_max_steps:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            steps += 1
        avg_reward += ep_reward
    return avg_reward / float(eval_episodes)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    ckpt_dir: Path,
    filename: str,
    payload: Dict[str, Any],
) -> Path:
    ensure_dir(ckpt_dir)
    path = ckpt_dir / filename
    torch.save(payload, path)
    return path


def log_checkpoint_artifact(
    cfg: "ExperimentConfig",
    ckpt_path: Path,
    *,
    aliases: Tuple[str, ...],
    metadata: Dict[str, Any],
) -> None:
    """
    Upload a checkpoint file to W&B Artifacts.

    This keeps large binary files out of Git while still versioning them in W&B.
    """
    if not cfg.use_wandb or not getattr(cfg, "wandb_artifacts", False):
        return
    if wandb is None or wandb.run is None:
        return

    # One artifact "stream" per run; each upload creates a new artifact version.
    art_name = f"{wandb.run.name}-ckpt"
    artifact = wandb.Artifact(name=art_name, type="model", metadata=metadata)
    artifact.add_file(str(ckpt_path), name=ckpt_path.name)
    wandb.log_artifact(artifact, aliases=list(aliases))


@dataclass
class ExperimentConfig:
    algo: str = "sac"  # sac | mbpo | dreamer
    env_name: str = ENV_NAME_DEFAULT
    seeds: Tuple[int, ...] = (1, 2, 3)

    device: Optional[str] = None

    # Training
    train_max_steps: int = TRAIN_MAX_STEPS_DEFAULT
    episode_max_steps: int = EPISODE_MAX_STEPS_DEFAULT
    replay_prefill_steps: int = REPLAY_PREFILL_STEPS_DEFAULT
    batch_size: int = BATCH_SIZE_DEFAULT

    # Evaluation
    eval_frequency_steps: int = EVAL_FREQUENCY_STEPS_DEFAULT
    eval_episodes_during_train: int = EVAL_EPISODES_DURING_TRAIN_DEFAULT
    eval_episodes_final: int = EVAL_EPISODES_FINAL_DEFAULT
    asymptotic_window_evals: int = ASYMPTOTIC_WINDOW_EVALS_DEFAULT
    convergence_threshold: float = CONVERGENCE_THRESHOLD_DEFAULT
    convergence_stable_k: int = CONVERGENCE_STABLE_K_DEFAULT

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_steps: int = CHECKPOINT_EVERY_STEPS_DEFAULT

    # Algorithm-specific (placeholders for MBPO/Dreamer)
    horizon: int = 5
    dreamer_arch: str = "MLP"
    mbpo_model_train_steps_per_env_step: int = 1
    mbpo_synthetic_updates_per_env_step: int = 1
    mbpo_ensemble_size: int = 7
    mbpo_top_k: int = 5

    dreamer_seq_len: int = DREAMER_SEQ_LEN_DEFAULT

    # Dreamer extras (paper-shaped stabilizers / likelihoods)
    dreamer_kl_beta: float = 1.0
    dreamer_auto_kl_beta: bool = True
    dreamer_target_kl: float = 3.0
    dreamer_kl_beta_lr: float = 1e-4
    dreamer_kl_beta_min: float = 0.0
    dreamer_kl_beta_max: float = 10.0
    dreamer_continuation_beta: float = 1.0
    dreamer_continuation_target: str = "episode_end"
    dreamer_free_nats: float = 3.0
    dreamer_kl_balance: float = 0.8
    dreamer_obs_std: float = 0.1
    dreamer_reward_std: float = 1.0
    dreamer_learn_std: bool = True
    dreamer_log_std_min: float = -5.0
    dreamer_log_std_max: float = 2.0
    dreamer_target_tau: float = 0.01
    dreamer_lambda: float = 0.95
    dreamer_actor_entropy_coef: float = 0.0

    # SAC extras
    sac_auto_alpha: bool = False
    sac_target_entropy: Optional[float] = None

    # W&B
    use_wandb: bool = True
    wandb_project: str = "CS5180_Pendulum_MBRL"
    wandb_entity: Optional[str] = None
    wandb_tags: Tuple[str, ...] = ()
    wandb_group: Optional[str] = None
    wandb_artifacts: bool = False


def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, default="sac", choices=["sac", "mbpo", "dreamer"])
    p.add_argument("--env", type=str, default=ENV_NAME_DEFAULT)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--train-steps", type=int, default=TRAIN_MAX_STEPS_DEFAULT)
    p.add_argument("--episode-max-steps", type=int, default=EPISODE_MAX_STEPS_DEFAULT)
    p.add_argument("--prefill-steps", type=int, default=REPLAY_PREFILL_STEPS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)

    p.add_argument("--eval-freq", type=int, default=EVAL_FREQUENCY_STEPS_DEFAULT)
    p.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES_DURING_TRAIN_DEFAULT)
    p.add_argument("--eval-episodes-final", type=int, default=EVAL_EPISODES_FINAL_DEFAULT)
    p.add_argument("--asymptotic-window", type=int, default=ASYMPTOTIC_WINDOW_EVALS_DEFAULT)
    p.add_argument("--threshold", type=float, default=CONVERGENCE_THRESHOLD_DEFAULT)
    p.add_argument("--stable-k", type=int, default=CONVERGENCE_STABLE_K_DEFAULT)

    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY_STEPS_DEFAULT)

    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--dreamer-arch", type=str, default="MLP")
    p.add_argument("--mbpo-model-steps", type=int, default=1)
    p.add_argument("--mbpo-synth-updates", type=int, default=1)
    p.add_argument("--mbpo-ensemble-size", type=int, default=7)
    p.add_argument("--mbpo-top-k", type=int, default=5)
    p.add_argument("--dreamer-seq-len", type=int, default=DREAMER_SEQ_LEN_DEFAULT)

    # Dreamer extras (match DreamerConfig fields)
    dreamer_defaults = DreamerConfig()
    p.add_argument("--dreamer-kl-beta", type=float, default=float(dreamer_defaults.kl_beta))
    p.add_argument(
        "--dreamer-auto-kl-beta",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(dreamer_defaults, "auto_kl_beta", True)),
    )
    p.add_argument("--dreamer-target-kl", type=float, default=float(getattr(dreamer_defaults, "target_kl", 3.0)))
    p.add_argument("--dreamer-kl-beta-lr", type=float, default=float(getattr(dreamer_defaults, "kl_beta_lr", 1e-4)))
    p.add_argument("--dreamer-kl-beta-min", type=float, default=float(getattr(dreamer_defaults, "kl_beta_min", 0.0)))
    p.add_argument("--dreamer-kl-beta-max", type=float, default=float(getattr(dreamer_defaults, "kl_beta_max", 10.0)))
    p.add_argument(
        "--dreamer-cont-beta",
        "--dreamer-continuation-beta",
        dest="dreamer_continuation_beta",
        type=float,
        default=float(getattr(dreamer_defaults, "continuation_beta", 1.0)),
    )
    p.add_argument(
        "--dreamer-cont-target",
        dest="dreamer_continuation_target",
        type=str,
        choices=("episode_end", "terminated"),
        default=str(getattr(dreamer_defaults, "continuation_target", "episode_end")),
        help="Which end signal trains continuation head: 'episode_end' (terminated|truncated) or 'terminated' only.",
    )
    p.add_argument("--dreamer-free-nats", type=float, default=float(dreamer_defaults.free_nats))
    p.add_argument("--dreamer-kl-balance", type=float, default=float(dreamer_defaults.kl_balance))
    p.add_argument("--dreamer-obs-std", type=float, default=float(dreamer_defaults.obs_std))
    p.add_argument("--dreamer-reward-std", type=float, default=float(dreamer_defaults.reward_std))
    p.add_argument(
        "--dreamer-learn-std",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(dreamer_defaults, "learn_std", True)),
    )
    p.add_argument("--dreamer-log-std-min", type=float, default=float(getattr(dreamer_defaults, "log_std_min", -5.0)))
    p.add_argument("--dreamer-log-std-max", type=float, default=float(getattr(dreamer_defaults, "log_std_max", 2.0)))
    p.add_argument("--dreamer-target-tau", type=float, default=float(dreamer_defaults.target_tau))
    p.add_argument("--dreamer-lambda", type=float, default=float(dreamer_defaults.lambda_))
    p.add_argument(
        "--dreamer-actor-entropy-coef",
        type=float,
        default=float(getattr(dreamer_defaults, "actor_entropy_coef", 0.0)),
    )

    # SAC extras
    p.add_argument("--sac-auto-alpha", action="store_true")
    p.add_argument("--sac-target-entropy", type=float, default=None)

    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="CS5180_Pendulum_MBRL")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="")
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-artifacts", action="store_true")

    args = p.parse_args()
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    tags = tuple(t.strip() for t in args.wandb_tags.split(",") if t.strip())
    return ExperimentConfig(
        algo=args.algo,
        env_name=args.env,
        seeds=seeds,
        device=args.device,
        train_max_steps=args.train_steps,
        episode_max_steps=args.episode_max_steps,
        replay_prefill_steps=args.prefill_steps,
        batch_size=args.batch_size,
        eval_frequency_steps=args.eval_freq,
        eval_episodes_during_train=args.eval_episodes,
        eval_episodes_final=args.eval_episodes_final,
        asymptotic_window_evals=args.asymptotic_window,
        convergence_threshold=args.threshold,
        convergence_stable_k=args.stable_k,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every_steps=args.checkpoint_every,
        horizon=args.horizon,
        dreamer_arch=args.dreamer_arch,
        mbpo_model_train_steps_per_env_step=args.mbpo_model_steps,
        mbpo_synthetic_updates_per_env_step=args.mbpo_synth_updates,
        mbpo_ensemble_size=args.mbpo_ensemble_size,
        mbpo_top_k=args.mbpo_top_k,
        dreamer_seq_len=args.dreamer_seq_len,
        dreamer_kl_beta=float(args.dreamer_kl_beta),
        dreamer_auto_kl_beta=bool(args.dreamer_auto_kl_beta),
        dreamer_target_kl=float(args.dreamer_target_kl),
        dreamer_kl_beta_lr=float(args.dreamer_kl_beta_lr),
        dreamer_kl_beta_min=float(args.dreamer_kl_beta_min),
        dreamer_kl_beta_max=float(args.dreamer_kl_beta_max),
        dreamer_continuation_beta=float(args.dreamer_continuation_beta),
        dreamer_continuation_target=str(args.dreamer_continuation_target),
        dreamer_free_nats=float(args.dreamer_free_nats),
        dreamer_kl_balance=float(args.dreamer_kl_balance),
        dreamer_obs_std=float(args.dreamer_obs_std),
        dreamer_reward_std=float(args.dreamer_reward_std),
        dreamer_learn_std=bool(args.dreamer_learn_std),
        dreamer_log_std_min=float(args.dreamer_log_std_min),
        dreamer_log_std_max=float(args.dreamer_log_std_max),
        dreamer_target_tau=float(args.dreamer_target_tau),
        dreamer_lambda=float(args.dreamer_lambda),
        dreamer_actor_entropy_coef=float(args.dreamer_actor_entropy_coef),
        sac_auto_alpha=bool(args.sac_auto_alpha),
        sac_target_entropy=args.sac_target_entropy,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_tags=tags,
        wandb_group=args.wandb_group,
        wandb_artifacts=bool(args.wandb_artifacts),
    )


def init_wandb(cfg: ExperimentConfig, seed: int) -> Any:
    if not cfg.use_wandb:
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed but use_wandb=True. Install wandb or pass --no-wandb.")

    run_name = make_run_name(cfg.algo, seed, horizon=cfg.horizon, arch=cfg.dreamer_arch)
    config_dict = dataclasses.asdict(cfg)
    config_dict["seed"] = seed
    config_dict["run_name"] = run_name
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        group=cfg.wandb_group,
        tags=list(cfg.wandb_tags),
        config=config_dict,
    )


def run_sac(cfg: ExperimentConfig, seed: int) -> Dict[str, Any]:
    device = pick_device(cfg.device)
    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    sac_cfg = SACConfig(
        action_scale=action_scale,
        action_bias=action_bias,
        auto_alpha=cfg.sac_auto_alpha,
        target_entropy=cfg.sac_target_entropy,
    )
    agent = SACAgent(state_dim, action_dim, device=device, cfg=sac_cfg)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Trackers
    eval_steps = []
    eval_returns = []
    convergence_step = None
    convergence_step_stable_k = None
    consecutive_above = 0

    obs, _ = env.reset(seed=seed)
    ep_steps = 0

    t0 = time.time()
    ckpt_root = Path(cfg.checkpoint_dir) / make_run_name(cfg.algo, seed, horizon=None)
    print(f"[SAC seed={seed}] device={device} train_steps={cfg.train_max_steps}")

    # prefill the replay buffer with random actions and then train the agent using policy updates
    for step in range(cfg.train_max_steps):
        if step < cfg.replay_prefill_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        # IMPORTANT: for learning/bootstrapping, treat only true environment termination as done.
        # Time-limit truncation should NOT cut off bootstrapping targets.
        done_for_learning = float(terminated)
        episode_end = terminated or truncated

        replay_buffer.add(obs, action, reward, next_obs, done_for_learning, episode_end=float(episode_end))

        obs = next_obs
        ep_steps += 1

        # after prefill, train the agent using policy updates
        # minibatch from the replay buffer
        if step >= cfg.replay_prefill_steps:
            loss_dict = agent.train(replay_buffer, batch_size=cfg.batch_size)
        else:
            loss_dict = {}

        if episode_end or ep_steps >= cfg.episode_max_steps:
            obs, _ = env.reset()
            ep_steps = 0

        # periodic evaluation
        if step % cfg.eval_frequency_steps == 0 and step >= cfg.replay_prefill_steps:
            eval_return = evaluate_policy(
                agent,
                env_name=cfg.env_name,
                eval_episodes=cfg.eval_episodes_during_train,
                seed=seed,
                episode_max_steps=cfg.episode_max_steps,
            )
            eval_steps.append(step)
            eval_returns.append(eval_return)

            elapsed_s = time.time() - t0
            steps_per_sec = (step + 1) / max(elapsed_s, 1e-8)
            remaining_steps = cfg.train_max_steps - (step + 1)
            eta_s = remaining_steps / max(steps_per_sec, 1e-8)
            progress = 100.0 * (step + 1) / float(cfg.train_max_steps)

            if wandb and wandb.run is not None:
                wandb.log(
                    {
                        "train/step": step,
                        "eval/return": eval_return,
                        "time/elapsed_s": elapsed_s,
                        "time/steps_per_sec": steps_per_sec,
                        "time/eta_s": eta_s,
                        **loss_dict,
                    },
                    step=step,
                )
            else:
                print(
                    f"[SAC seed={seed}] "
                    f"step={step}/{cfg.train_max_steps} ({progress:.1f}%) "
                    f"eval_return={eval_return:.2f} "
                    f"speed={steps_per_sec:.1f} step/s "
                    f"ETA={format_eta(eta_s)}"
                )

            # convergence checkpoint
            if eval_return > cfg.convergence_threshold and convergence_step is None:
                convergence_step = step
                payload = {
                    "algo": "sac",
                    "seed": seed,
                    "step": step,
                    "env_name": cfg.env_name,
                    "agent": agent.get_state(),
                }
                ckpt_path = save_checkpoint(
                    ckpt_root, f"converged_{abs(int(cfg.convergence_threshold))}.pt", payload
                )

            # stable convergence: first step where we have K consecutive eval points above threshold
            if eval_return > cfg.convergence_threshold:
                consecutive_above += 1
            else:
                consecutive_above = 0
            if (
                convergence_step_stable_k is None
                and consecutive_above >= int(cfg.convergence_stable_k)
            ):
                convergence_step_stable_k = step
                log_checkpoint_artifact(
                    cfg,
                    ckpt_path,
                    aliases=("latest", f"converged-{abs(int(cfg.convergence_threshold))}"),
                    metadata={
                        "algo": "sac",
                        "seed": seed,
                        "step": step,
                        "env_name": cfg.env_name,
                        "threshold": cfg.convergence_threshold,
                    },
                )

        # regular step checkpoint
        if cfg.checkpoint_every_steps and step > 0 and step % cfg.checkpoint_every_steps == 0:
            payload = {
                "algo": "sac",
                "seed": seed,
                "step": step,
                "env_name": cfg.env_name,
                "agent": agent.get_state(),
            }
            ckpt_path = save_checkpoint(ckpt_root, f"step_{format_step_k(step)}.pt", payload)
            log_checkpoint_artifact(
                cfg,
                ckpt_path,
                aliases=("latest", f"step-{format_step_k(step)}"),
                metadata={"algo": "sac", "seed": seed, "step": step, "env_name": cfg.env_name},
            )

    # final checkpoint
    payload = {
        "algo": "sac",
        "seed": seed,
        "step": cfg.train_max_steps,
        "env_name": cfg.env_name,
        "agent": agent.get_state(),
    }
    ckpt_path = save_checkpoint(ckpt_root, "final_stage.pt", payload)
    log_checkpoint_artifact(
        cfg,
        ckpt_path,
        aliases=("latest", "final"),
        metadata={"algo": "sac", "seed": seed, "step": cfg.train_max_steps, "env_name": cfg.env_name},
    )

    # final evaluation summary
    final_return_mean = evaluate_policy(
        agent,
        env_name=cfg.env_name,
        eval_episodes=cfg.eval_episodes_final,
        seed=seed,
        episode_max_steps=cfg.episode_max_steps,
    )
    asymptotic = None
    if len(eval_returns) > 0:
        window = min(cfg.asymptotic_window_evals, len(eval_returns))
        asymptotic = float(np.mean(eval_returns[-window:]))
        if wandb and wandb.run is not None:
            wandb.summary["eval/asymptotic_return"] = asymptotic
            wandb.summary["eval/convergence_step"] = convergence_step
            wandb.summary["eval/final_return_mean"] = float(final_return_mean)
            wandb.summary["eval/convergence_step_stable_k"] = convergence_step_stable_k
            # Also log as a final history point for convenience.
            wandb.log(
                {
                    "train/step": cfg.train_max_steps,
                    "eval/final_return_mean": float(final_return_mean),
                    "eval/convergence_step_stable_k": convergence_step_stable_k,
                },
                step=cfg.train_max_steps,
            )

    return {
        "seed": seed,
        "eval_steps": eval_steps,
        "eval_returns": eval_returns,
        "asymptotic_return": asymptotic,
        "convergence_step": convergence_step,
        "final_return_mean": float(final_return_mean),
        "convergence_step_stable_k": convergence_step_stable_k,
    }


def run_mbpo(cfg: ExperimentConfig, seed: int) -> Dict[str, Any]:
    device = pick_device(cfg.device)
    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    mbpo_sac_cfg = SACConfig(
        action_scale=action_scale,
        action_bias=action_bias,
        auto_alpha=cfg.sac_auto_alpha,
        target_entropy=cfg.sac_target_entropy,
    )
    agent = MBPOAgent(
        state_dim,
        action_dim,
        device=device,
        sac_cfg=mbpo_sac_cfg,
        mbpo_cfg=MBPOConfig(
            horizon=cfg.horizon,
            model_ensemble_size=cfg.mbpo_ensemble_size,
            model_top_k=cfg.mbpo_top_k,
            model_train_steps_per_env_step=cfg.mbpo_model_train_steps_per_env_step,
            synthetic_updates_per_env_step=cfg.mbpo_synthetic_updates_per_env_step,
        ),
    )
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    eval_steps = []
    eval_returns = []
    convergence_step = None
    convergence_step_stable_k = None
    consecutive_above = 0

    obs, _ = env.reset(seed=seed)
    ep_steps = 0

    t0 = time.time()
    ckpt_root = Path(cfg.checkpoint_dir) / make_run_name(cfg.algo, seed, horizon=cfg.horizon)

    for step in range(cfg.train_max_steps):
        if step < cfg.replay_prefill_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done_for_learning = float(terminated)
        episode_end = terminated or truncated

        replay_buffer.add(obs, action, reward, next_obs, done_for_learning, episode_end=float(episode_end))
        obs = next_obs
        ep_steps += 1

        # Updates (after warm-up)
        loss_dict: Dict[str, float] = {}
        if step >= cfg.replay_prefill_steps:
            # real SAC update
            loss_dict.update(agent.train_policy_on_real(replay_buffer, batch_size=cfg.batch_size))

            # model updates
            model_stats = agent.train_model(replay_buffer, batch_size=cfg.batch_size)
            loss_dict.update(
                {
                    "model/loss": model_stats.loss,
                    "model/nll": model_stats.nll,
                    "model/terminated_bce": getattr(model_stats, "terminated_bce", 0.0),
                    "model/terminated_acc": getattr(model_stats, "terminated_acc", 0.0),
                    "model/terminated_rate": getattr(model_stats, "terminated_rate", 0.0),
                    "model/mse_next_state": model_stats.mse_next_state,
                    "model/mse_reward": model_stats.mse_reward,
                    "model/avg_log_std": model_stats.avg_log_std,
                    "model/epistemic_var_next_state": model_stats.epistemic_var_next_state,
                    "model/epistemic_var_reward": model_stats.epistemic_var_reward,
                }
            )
            # Elite indices are a list; log as a string for readability.
            loss_dict["model/selected_model_indices"] = str(model_stats.selected_model_indices)

            # synthetic SAC updates
            syn_losses = agent.train_policy_on_synthetic(replay_buffer, batch_size=cfg.batch_size)
            loss_dict.update({f"synthetic/{k}": v for k, v in syn_losses.items()})

        if episode_end or ep_steps >= cfg.episode_max_steps:
            obs, _ = env.reset()
            ep_steps = 0

        if step % cfg.eval_frequency_steps == 0 and step >= cfg.replay_prefill_steps:
            eval_return = evaluate_policy(
                agent,
                env_name=cfg.env_name,
                eval_episodes=cfg.eval_episodes_during_train,
                seed=seed,
                episode_max_steps=cfg.episode_max_steps,
            )
            eval_steps.append(step)
            eval_returns.append(eval_return)

            if wandb and wandb.run is not None:
                wandb.log(
                    {
                        "train/step": step,
                        "eval/return": eval_return,
                        "time/elapsed_s": time.time() - t0,
                        **loss_dict,
                    },
                    step=step,
                )
            else:
                print(
                    f"[{cfg.algo.upper()} seed={seed} horizon={cfg.horizon}] "
                    f"step={step} eval_return={eval_return:.2f}"
                )

            if eval_return > cfg.convergence_threshold and convergence_step is None:
                convergence_step = step
                payload = {
                    "algo": "mbpo",
                    "seed": seed,
                    "step": step,
                    "env_name": cfg.env_name,
                    "horizon": cfg.horizon,
                    "agent": agent.get_state(),
                }
                ckpt_path = save_checkpoint(
                    ckpt_root, f"converged_{abs(int(cfg.convergence_threshold))}.pt", payload
                )

            # stable convergence: first step where we have K consecutive eval points above threshold
            if eval_return > cfg.convergence_threshold:
                consecutive_above += 1
            else:
                consecutive_above = 0
            if (
                convergence_step_stable_k is None
                and consecutive_above >= int(cfg.convergence_stable_k)
            ):
                convergence_step_stable_k = step
                log_checkpoint_artifact(
                    cfg,
                    ckpt_path,
                    aliases=("latest", f"converged-{abs(int(cfg.convergence_threshold))}"),
                    metadata={
                        "algo": "mbpo",
                        "seed": seed,
                        "step": step,
                        "env_name": cfg.env_name,
                        "horizon": cfg.horizon,
                        "threshold": cfg.convergence_threshold,
                    },
                )

        if cfg.checkpoint_every_steps and step > 0 and step % cfg.checkpoint_every_steps == 0:
            payload = {
                "algo": "mbpo",
                "seed": seed,
                "step": step,
                "env_name": cfg.env_name,
                "horizon": cfg.horizon,
                "agent": agent.get_state(),
            }
            ckpt_path = save_checkpoint(ckpt_root, f"step_{format_step_k(step)}.pt", payload)
            log_checkpoint_artifact(
                cfg,
                ckpt_path,
                aliases=("latest", f"step-{format_step_k(step)}"),
                metadata={
                    "algo": "mbpo",
                    "seed": seed,
                    "step": step,
                    "env_name": cfg.env_name,
                    "horizon": cfg.horizon,
                },
            )

    payload = {
        "algo": "mbpo",
        "seed": seed,
        "step": cfg.train_max_steps,
        "env_name": cfg.env_name,
        "horizon": cfg.horizon,
        "agent": agent.get_state(),
    }
    ckpt_path = save_checkpoint(ckpt_root, "final_stage.pt", payload)
    log_checkpoint_artifact(
        cfg,
        ckpt_path,
        aliases=("latest", "final"),
        metadata={
            "algo": "mbpo",
            "seed": seed,
            "step": cfg.train_max_steps,
            "env_name": cfg.env_name,
            "horizon": cfg.horizon,
        },
    )

    final_return_mean = evaluate_policy(
        agent,
        env_name=cfg.env_name,
        eval_episodes=cfg.eval_episodes_final,
        seed=seed,
        episode_max_steps=cfg.episode_max_steps,
    )

    asymptotic = None
    if len(eval_returns) > 0:
        window = min(cfg.asymptotic_window_evals, len(eval_returns))
        asymptotic = float(np.mean(eval_returns[-window:]))
        if wandb and wandb.run is not None:
            wandb.summary["eval/asymptotic_return"] = asymptotic
            wandb.summary["eval/convergence_step"] = convergence_step
            wandb.summary["eval/final_return_mean"] = float(final_return_mean)
            wandb.summary["eval/convergence_step_stable_k"] = convergence_step_stable_k
            wandb.log(
                {
                    "train/step": cfg.train_max_steps,
                    "eval/final_return_mean": float(final_return_mean),
                    "eval/convergence_step_stable_k": convergence_step_stable_k,
                },
                step=cfg.train_max_steps,
            )

    return {
        "seed": seed,
        "eval_steps": eval_steps,
        "eval_returns": eval_returns,
        "asymptotic_return": asymptotic,
        "convergence_step": convergence_step,
        "final_return_mean": float(final_return_mean),
        "convergence_step_stable_k": convergence_step_stable_k,
    }


def run_dreamer(cfg: ExperimentConfig, seed: int) -> Dict[str, Any]:
    device = pick_device(cfg.device)
    env = gym.make(cfg.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    agent = DreamerAgent(
        obs_dim,
        action_dim,
        action_scale=action_scale,
        action_bias=action_bias,
        device=device,
        cfg=DreamerConfig(
            horizon=cfg.horizon,
            kl_beta=cfg.dreamer_kl_beta,
            auto_kl_beta=cfg.dreamer_auto_kl_beta,
            target_kl=cfg.dreamer_target_kl,
            kl_beta_lr=cfg.dreamer_kl_beta_lr,
            kl_beta_min=cfg.dreamer_kl_beta_min,
            kl_beta_max=cfg.dreamer_kl_beta_max,
            continuation_beta=cfg.dreamer_continuation_beta,
            continuation_target=cfg.dreamer_continuation_target,
            free_nats=cfg.dreamer_free_nats,
            kl_balance=cfg.dreamer_kl_balance,
            obs_std=cfg.dreamer_obs_std,
            reward_std=cfg.dreamer_reward_std,
            learn_std=cfg.dreamer_learn_std,
            log_std_min=cfg.dreamer_log_std_min,
            log_std_max=cfg.dreamer_log_std_max,
            target_tau=cfg.dreamer_target_tau,
            lambda_=cfg.dreamer_lambda,
            actor_entropy_coef=cfg.dreamer_actor_entropy_coef,
        ),
    )
    replay_buffer = ReplayBuffer(obs_dim, action_dim)

    eval_steps = []
    eval_returns = []
    convergence_step = None
    convergence_step_stable_k = None
    consecutive_above = 0

    obs, _ = env.reset(seed=seed)
    agent.reset_episode()
    ep_steps = 0

    t0 = time.time()
    ckpt_root = Path(cfg.checkpoint_dir) / make_run_name(cfg.algo, seed, horizon=cfg.horizon, arch=cfg.dreamer_arch)

    for step in range(cfg.train_max_steps):
        if step < cfg.replay_prefill_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done_for_learning = float(terminated)
        episode_end = terminated or truncated
        replay_buffer.add(obs, action, reward, next_obs, done_for_learning, episode_end=float(episode_end))
        obs = next_obs
        ep_steps += 1

        loss_dict: Dict[str, float] = {}
        if step >= cfg.replay_prefill_steps and replay_buffer.size >= cfg.dreamer_seq_len + 1:
            # Train world model on sequences
            obs_seq, act_seq, rew_seq, _, done_seq, episode_end_seq = replay_buffer.sample_sequences(
                cfg.batch_size, cfg.dreamer_seq_len
            )
            obs_seq = obs_seq.to(agent.device)
            act_seq = act_seq.to(agent.device)
            rew_seq = rew_seq.to(agent.device)
            episode_end_seq = episode_end_seq.to(agent.device)
            wm_losses = agent.train_world_model(
                obs_seq, act_seq, reward_seq=rew_seq, terminated_seq=done_seq, episode_end_seq=episode_end_seq
            )
            loss_dict.update(wm_losses)

            ac_losses = agent.train_actor_critic(obs_seq, action_seq=act_seq)
            loss_dict.update(ac_losses)

        if episode_end or ep_steps >= cfg.episode_max_steps:
            obs, _ = env.reset()
            agent.reset_episode()
            ep_steps = 0

        if step % cfg.eval_frequency_steps == 0 and step >= cfg.replay_prefill_steps:
            eval_return = evaluate_policy(
                agent,
                env_name=cfg.env_name,
                eval_episodes=cfg.eval_episodes_during_train,
                seed=seed,
                episode_max_steps=cfg.episode_max_steps,
            )
            eval_steps.append(step)
            eval_returns.append(eval_return)

            if wandb and wandb.run is not None:
                wandb.log(
                    {
                        "train/step": step,
                        "eval/return": eval_return,
                        "time/elapsed_s": time.time() - t0,
                        **loss_dict,
                    },
                    step=step,
                )
            else:
                print(
                    f"[{cfg.algo.upper()} seed={seed} horizon={cfg.horizon}] "
                    f"step={step} eval_return={eval_return:.2f}"
                )

            if eval_return > cfg.convergence_threshold and convergence_step is None:
                convergence_step = step
                payload = {
                    "algo": "dreamer",
                    "seed": seed,
                    "step": step,
                    "env_name": cfg.env_name,
                    "horizon": cfg.horizon,
                    "agent": agent.get_state(),
                }
                ckpt_path = save_checkpoint(
                    ckpt_root, f"converged_{abs(int(cfg.convergence_threshold))}.pt", payload
                )

            # stable convergence: first step where we have K consecutive eval points above threshold
            if eval_return > cfg.convergence_threshold:
                consecutive_above += 1
            else:
                consecutive_above = 0
            if (
                convergence_step_stable_k is None
                and consecutive_above >= int(cfg.convergence_stable_k)
            ):
                convergence_step_stable_k = step
                log_checkpoint_artifact(
                    cfg,
                    ckpt_path,
                    aliases=("latest", f"converged-{abs(int(cfg.convergence_threshold))}"),
                    metadata={
                        "algo": "dreamer",
                        "seed": seed,
                        "step": step,
                        "env_name": cfg.env_name,
                        "horizon": cfg.horizon,
                        "threshold": cfg.convergence_threshold,
                    },
                )

        if cfg.checkpoint_every_steps and step > 0 and step % cfg.checkpoint_every_steps == 0:
            payload = {
                "algo": "dreamer",
                "seed": seed,
                "step": step,
                "env_name": cfg.env_name,
                "horizon": cfg.horizon,
                "agent": agent.get_state(),
            }
            ckpt_path = save_checkpoint(ckpt_root, f"step_{format_step_k(step)}.pt", payload)
            log_checkpoint_artifact(
                cfg,
                ckpt_path,
                aliases=("latest", f"step-{format_step_k(step)}"),
                metadata={
                    "algo": "dreamer",
                    "seed": seed,
                    "step": step,
                    "env_name": cfg.env_name,
                    "horizon": cfg.horizon,
                },
            )

    payload = {
        "algo": "dreamer",
        "seed": seed,
        "step": cfg.train_max_steps,
        "env_name": cfg.env_name,
        "horizon": cfg.horizon,
        "agent": agent.get_state(),
    }
    ckpt_path = save_checkpoint(ckpt_root, "final_stage.pt", payload)
    log_checkpoint_artifact(
        cfg,
        ckpt_path,
        aliases=("latest", "final"),
        metadata={
            "algo": "dreamer",
            "seed": seed,
            "step": cfg.train_max_steps,
            "env_name": cfg.env_name,
            "horizon": cfg.horizon,
        },
    )

    final_return_mean = evaluate_policy(
        agent,
        env_name=cfg.env_name,
        eval_episodes=cfg.eval_episodes_final,
        seed=seed,
        episode_max_steps=cfg.episode_max_steps,
    )

    asymptotic = None
    if len(eval_returns) > 0:
        window = min(cfg.asymptotic_window_evals, len(eval_returns))
        asymptotic = float(np.mean(eval_returns[-window:]))
        if wandb and wandb.run is not None:
            wandb.summary["eval/asymptotic_return"] = asymptotic
            wandb.summary["eval/convergence_step"] = convergence_step
            wandb.summary["eval/final_return_mean"] = float(final_return_mean)
            wandb.summary["eval/convergence_step_stable_k"] = convergence_step_stable_k
            wandb.log(
                {
                    "train/step": cfg.train_max_steps,
                    "eval/final_return_mean": float(final_return_mean),
                    "eval/convergence_step_stable_k": convergence_step_stable_k,
                },
                step=cfg.train_max_steps,
            )

    return {
        "seed": seed,
        "eval_steps": eval_steps,
        "eval_returns": eval_returns,
        "asymptotic_return": asymptotic,
        "convergence_step": convergence_step,
        "final_return_mean": float(final_return_mean),
        "convergence_step_stable_k": convergence_step_stable_k,
    }


def main() -> None:
    cfg = parse_args()
    algo = cfg.algo.lower()

    all_results = []
    for seed in cfg.seeds:
        run = init_wandb(cfg, seed)
        try:
            if algo == "sac":
                result = run_sac(cfg, seed)
            elif algo == "mbpo":
                result = run_mbpo(cfg, seed)
            elif algo == "dreamer":
                result = run_dreamer(cfg, seed)
            else:
                raise ValueError(f"Unknown algo: {algo}")
            all_results.append(result)
        finally:
            if run is not None and wandb is not None:
                wandb.finish()

    # Print aggregate summary (mean ± std over seeds)
    asymptotics = [r["asymptotic_return"] for r in all_results if r["asymptotic_return"] is not None]
    if asymptotics:
        mean_perf = float(np.mean(asymptotics))
        std_perf = float(np.std(asymptotics))
        print(f"[Summary] {cfg.algo.upper()} Asymptotic Return: {mean_perf:.2f} ± {std_perf:.2f}")


if __name__ == "__main__":
    main()


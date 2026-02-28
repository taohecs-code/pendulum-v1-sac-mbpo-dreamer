"""
Soft Actor-Critic (SAC) Agent.

This class combines the Actor and Critic networks defined in `sac_networks`.
It strictly implements the core mechanisms required by the SAC algorithm, including:
- Twin-Q target computation to mitigate overestimation bias.
- Exponential Moving Average (EMA) for soft updates of the target critic network.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sac_networks import Actor, Critic


@dataclass
class SACConfig:
    # RL hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    # Entropy temperature (alpha)
    alpha: float = 0.2
    auto_alpha: bool = False
    target_entropy: float | None = None  # if None, default to -|A|

    # Action mapping: a = tanh(u) * scale + bias
    action_scale: float | torch.Tensor = 1.0
    action_bias: float | torch.Tensor = 0.0




class SACAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        action_scale: float | torch.Tensor = 1.0,
        action_bias: float | torch.Tensor = 0.0,
        device=None,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha: bool = False,
        target_entropy: float | None = None,
        cfg: SACConfig | None = None,
    ):
        """
        Initialize the SAC agent.
        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.
            lr (float): The learning rate.
            gamma (float): The discount factor.
            tau (float): The soft update rate.
            alpha (float): The temperature parameter.
        """

        # Preferred: pass `cfg=SACConfig(...)`. Backward-compatible: construct config from legacy kwargs.
        if cfg is None:
            cfg = SACConfig(
                lr=float(lr),
                gamma=float(gamma),
                tau=float(tau),
                alpha=float(alpha),
                auto_alpha=bool(auto_alpha),
                target_entropy=(float(target_entropy) if target_entropy is not None else None),
                action_scale=action_scale,
                action_bias=action_bias,
            )
        self.cfg = cfg

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = self.cfg.action_scale
        self.action_bias = self.cfg.action_bias
        self.lr = float(self.cfg.lr)
        self.gamma = float(self.cfg.gamma)
        self.tau = float(self.cfg.tau)
        self.alpha = float(self.cfg.alpha)
        self.auto_alpha = bool(self.cfg.auto_alpha)

        # Common default in SAC-v2: target_entropy = -|A|
        self.target_entropy = (
            float(self.cfg.target_entropy) if self.cfg.target_entropy is not None else -float(action_dim)
        )

        self.actor = Actor(
            state_dim,
            action_dim,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
        ).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        # at first the target critic network is the same as the critic network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # create optimizers for the actor and critic networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # Optional: automatic entropy temperature tuning (SAC-v2)
        # optimize log_alpha to keep policy entropy close to target_entropy.
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                float(torch.log(torch.tensor(self.alpha))),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
    
    def select_action(self, state, deterministic: bool = False):
        """
        Select an action for the given state. No need to update the actor network.
        Args:
            state (torch.Tensor): The state. 
            
            deterministic (bool): Whether to use the deterministic action.
        Returns:
            action (torch.Tensor): The action.
        """

        self.actor.eval()

        with torch.no_grad():
            # NumPy (1D) -> Tensor (1D) -> Tensor (2D, Batch=1) -> Device
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _ = self.actor(state, deterministic)

        # -> cpu -> detach -> NumPy (1D)    
        return action.cpu().detach().numpy()[0] # action.cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=256):
        """
        Update the SAC agent.
        Args:
            replay_buffer (ReplayBuffer): The replay buffer.
            batch_size (int): The batch size.
        """
        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        return self.train_from_tensors(state, action, reward, next_state, done)

    def train_from_tensors(self, state, action, reward, next_state, done):
        """
        Update the SAC agent from an explicit batch of tensors.
        Tensors will be moved to the agent device if needed.
        Returns a dict of scalar losses for logging.
        """
        self.actor.train()
        self.critic.train()

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # compute the td target so do not update the critic network
        with torch.no_grad():
            # compute the next-state Q-values using the target critic network
            next_action, next_log_prob = self.actor(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            # min(Q1(s',a'), Q2(s',a')) - alpha * log_pi(a'|s')(entropy bonus)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            y = reward + self.gamma * (1 - done) * target_q

        
        # update the critic network
        current_q1, current_q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_q1, y.detach()) + F.mse_loss(current_q2, y.detach())
        
        self.critic_optimizer.zero_grad()
        
        critic_loss.backward()
        
        self.critic_optimizer.step()

        # update the actor network
        action, log_prob = self.actor(state)

        current_q1, current_q2 = self.critic(state, action)
        current_q = torch.min(current_q1, current_q2)

        actor_loss = torch.mean(self.alpha * log_prob - current_q)

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        # update alpha to match target entropy
        alpha_loss = None
        if self.auto_alpha and self.alpha_optimizer is not None and self.log_alpha is not None:
            # Use the current policy log_prob; detach so alpha update doesn't backprop into actor.
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = float(self.log_alpha.detach().exp().item())

        # ema update the target critic network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "loss/critic": float(critic_loss.item()),
            "loss/actor": float(actor_loss.item()),
            **({"loss/alpha": float(alpha_loss.item()), "alpha": float(self.alpha)} if alpha_loss is not None else {}),
        }

    def get_state(self):
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cfg": {
                "lr": float(self.lr),
                "gamma": float(self.gamma),
                "tau": float(self.tau),
                "alpha": float(self.alpha),
                "auto_alpha": bool(self.auto_alpha),
                # store the resolved value (could be auto default)
                "target_entropy": float(self.target_entropy),
                "action_scale": (
                    self.action_scale.detach().cpu().tolist()
                    if isinstance(self.action_scale, torch.Tensor)
                    else self.action_scale
                ),
                "action_bias": (
                    self.action_bias.detach().cpu().tolist()
                    if isinstance(self.action_bias, torch.Tensor)
                    else self.action_bias
                ),
            },
            "action_scale": (
                self.action_scale.detach().cpu().tolist()
                if isinstance(self.action_scale, torch.Tensor)
                else self.action_scale
            ),
            "action_bias": (
                self.action_bias.detach().cpu().tolist()
                if isinstance(self.action_bias, torch.Tensor)
                else self.action_bias
            ),
            "lr": self.lr,
            "gamma": self.gamma,
            "tau": self.tau,
            "alpha": self.alpha,
            "auto_alpha": self.auto_alpha,
            "target_entropy": self.target_entropy,
            "device": str(self.device),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": (self.log_alpha.detach().cpu().item() if self.log_alpha is not None else None),
            "alpha_optimizer": (self.alpha_optimizer.state_dict() if self.alpha_optimizer is not None else None),
        }

    def load_state(self, state):
        self.state_dim = state["state_dim"]
        self.action_dim = state["action_dim"]

        # Preferred path: load config if present. Backward-compatible with older checkpoints.
        cfg_dict = state.get("cfg", None)
        if isinstance(cfg_dict, dict):
            self.cfg = SACConfig(
                lr=float(cfg_dict.get("lr", 3e-4)),
                gamma=float(cfg_dict.get("gamma", 0.99)),
                tau=float(cfg_dict.get("tau", 0.005)),
                alpha=float(cfg_dict.get("alpha", 0.2)),
                auto_alpha=bool(cfg_dict.get("auto_alpha", False)),
                target_entropy=float(cfg_dict.get("target_entropy", -float(self.action_dim))),
                action_scale=cfg_dict.get("action_scale", 1.0),
                action_bias=cfg_dict.get("action_bias", 0.0),
            )
        else:
            self.cfg = SACConfig(
                lr=float(state.get("lr", 3e-4)),
                gamma=float(state.get("gamma", 0.99)),
                tau=float(state.get("tau", 0.005)),
                alpha=float(state.get("alpha", 0.2)),
                auto_alpha=bool(state.get("auto_alpha", False)),
                target_entropy=state.get("target_entropy", -float(self.action_dim)),
                action_scale=state.get("action_scale", 1.0),
                action_bias=state.get("action_bias", 0.0),
            )

        self.action_scale = self.cfg.action_scale
        self.action_bias = self.cfg.action_bias
        self.lr = float(self.cfg.lr)
        self.gamma = float(self.cfg.gamma)
        self.tau = float(self.cfg.tau)
        self.alpha = float(self.cfg.alpha)
        self.auto_alpha = bool(self.cfg.auto_alpha)
        self.target_entropy = (
            float(self.cfg.target_entropy) if self.cfg.target_entropy is not None else -float(self.action_dim)
        )

        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

        # restore alpha tuning state if present
        if self.auto_alpha:
            log_alpha_val = state.get("log_alpha", None)
            if log_alpha_val is not None:
                self.log_alpha = torch.tensor(
                    float(log_alpha_val),
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                self.alpha = float(self.log_alpha.detach().exp().item())
            else:
                # fall back to current alpha
                self.log_alpha = torch.tensor(
                    float(torch.log(torch.tensor(self.alpha))),
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=True,
                )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
            opt_state = state.get("alpha_optimizer", None)
            if opt_state is not None:
                self.alpha_optimizer.load_state_dict(opt_state)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
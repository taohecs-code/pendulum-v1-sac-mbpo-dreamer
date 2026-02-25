"""
Soft Actor-Critic (SAC) Agent.

This class combines the Actor and Critic networks defined in `sac_networks`. 
It strictly implements the core mechanisms required by the SAC algorithm, including:
- Twin-Q target computation to mitigate overestimation bias.
- Exponential Moving Average (EMA) for soft updates of the target critic network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sac_networks import Actor, Critic




class SACAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        device=None,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
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
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        # at first the target critic network is the same as the critic network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # create optimizers for the actor and critic networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
    
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

        # ema update the target critic network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "loss/critic": float(critic_loss.item()),
            "loss/actor": float(actor_loss.item()),
        }

    def get_state(self):
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "tau": self.tau,
            "alpha": self.alpha,
            "device": str(self.device),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state(self, state):
        self.state_dim = state["state_dim"]
        self.action_dim = state["action_dim"]
        self.lr = state["lr"]
        self.gamma = state["gamma"]
        self.tau = state["tau"]
        self.alpha = state["alpha"]

        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
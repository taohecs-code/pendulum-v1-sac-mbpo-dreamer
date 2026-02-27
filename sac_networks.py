"""
Soft Actor-Critic (SAC) implementation for continuous control tasks.
Reference: https://arxiv.org/pdf/1801.01290.pdf (and SAC-v2 refinements)

Pendulum-v1 Environment Specifications:
- State & Action Space: For Pendulum-v1, the Actor network's input state dimension 
  must be 3 (comprising cos(theta), sin(theta), and theta_dot). The output action 
  must be a 1D scalar, and the action range must be scaled/clipped to [-2.0, 2.0].

- Actor Network (Policy Improvement): 
  Must output a Gaussian policy and apply the reparameterization trick to enable 
  differentiable sampling. During the Evaluation phase, stochastic sampling must 
  be disabled, and the network should deterministically output the policy's mean.

- Critic Network (Policy Evaluation): 
  Must utilize a Twin-Q architecture (training two independent Q-networks 
  simultaneously and taking the minimum when computing the TD target) to mitigate 
  overestimation bias. A separate V-network is explicitly not required.

- Network Updates: 
  The target Critic networks must be smoothly updated using an Exponential Moving 
  Average (EMA, i.e., soft updates).

- Data Initialization: 
  Before formal training loops begin, the agent must interact with the environment 
  using uniformly random actions for 1,000 steps to warm up the Replay Buffer.
"""


import torch
import torch.nn as nn
from torch.distributions import Normal

# environment: Pendulum-v1
ACTION_BOUNDS =2.0 # (-2.0, 2.0)
ACTION_DIM = 1 # (torque)

STATE_DIM = 3 # (theta, theta_dot, omega)
HIDDEN_DIM = 256 # because the state dimension for the pendulum is 3, we need to use a higher dimension for hidden layers to capture the complexity of the state space


class Actor(nn.Module):
    """
        Actor Network for Soft Actor-Critic (SAC).

        Unlike deterministic policies (e.g., DDPG) that use argmax or output exact actions,
        the SAC Actor outputs a stochastic policy modeled as a Gaussian distribution.

        Key Mechanisms:
        1. Reparameterization Trick: 
        Samples are drawn in a differentiable way to allow gradients to flow back 
        from the Critic to the Actor.
        Formula: u = mean + std * epsilon, where epsilon ~ N(0, 1)

        2. Squashed Gaussian (tanh):
        The raw action 'u' is squashed into a bounded continuous range using tanh.
        Formula: a = tanh(u)

        3. Maximum Entropy RL:
        The network outputs the action and its log probability (log_pi). 
        The log_pi must be mathematically corrected for the tanh squashing.

        Optimization Details:
        - Objective: Maximize expected return AND policy entropy.
        Objective = E [ Q(s, a) - alpha * log_pi(a|s) ]
        
        - Loss Function: Since PyTorch minimizes, we negate the objective.
        The Actor is updated based on the Q-values provided by the frozen Critic.
        Loss = E [ alpha * log_pi(a|s) - min(Q1(s, a), Q2(s, a)) ]
        """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # need mean, variance and log_pi of the action distribution
        # but we can't directly output the variance(for relu will truncate the negative variance to zero), so we need to output the log_std and use exp to get the variance
        # calculate the log_pi later in the forward method(after sampling the action)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, deterministic: bool = False):
        """
        Forward pass of the actor network.
        Args:
            state (torch.Tensor): The state of the environment.
            deterministic (bool): Whether to use the deterministic action(for evaluation).
        Returns:
            action (torch.Tensor): The action.
            log_prob (torch.Tensor): The log probability of the action.
        """
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2) # for safety, usually the log_std is in the range of [-20, 2] for stability(avoid zero division and explosion)
        std = torch.exp(log_std)

        # if deterministic, use the mean as the action
        if deterministic:
            action = torch.tanh(mean) * ACTION_BOUNDS
            return action, None
        
        # if stochastic, sample the action from the distribution
        normal = Normal(mean, std)
        x_t = normal.rsample() # automatically applies the reparameterization trick
        
        y_t = torch.tanh(x_t) # bounded to [-1, 1]
        action = y_t * ACTION_BOUNDS # bounded to [-ACTION_BOUNDS, ACTION_BOUNDS]

        # calculate the log probability of the action
        log_prob = normal.log_prob(x_t)
        # Change of Variables Formula: log(pi(y)) = log(pi(x)) - log(|dy/dx|)
        # y is tanh(x), so dy/dx = 1 - y^2
        # 1e-6 is a small constant to avoid log(0)
        log_prob -= torch.log(ACTION_BOUNDS * (1 - y_t.pow(2)) + 1e-6)
        # sum over the action dimension and keep the batch dimension
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

class Critic(nn.Module):
    """
    Twin-Q Critic Network for Soft Actor-Critic (SAC).

    This class defines the two "online" (fast) Q-networks. Both networks are 
    updated simultaneously via gradient descent to mitigate overestimation bias.
    Note: The "target" (slow) networks are simply separate instances of this 
    class that do not compute losses; they are updated via Polyak (soft) averaging.

    Optimization Details:
    - TD Target (y): The "ground truth" value used for training the online critics.
    It incorporates the environment reward and the discounted future value 
    (including the maximum entropy bonus).
    Formula: y = r + gamma * (1 - done) * [min(Q1_target(s', a'), Q2_target(s', a')) - alpha * log_pi(a'|s')]
    
    - Objective (Loss): Minimize the Mean Squared Error (MSE) between the 
    online Q-predictions and the detached TD Target.
    Formula:
    Loss_Q1 = MSE( Q1_online(s, a), y.detach() )
    Loss_Q2 = MSE( Q2_online(s, a), y.detach() )
    """
        
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state, action):
        """
        Forward pass of the critic network.
        Args:
            state (torch.Tensor): The state of the environment.
            action (torch.Tensor): The action.
        Returns:
            q1_score (torch.Tensor): The Q-value of the first critic network.
            q2_score (torch.Tensor): The Q-value of the second critic network.
        """
        # concatenate the state and action as input (s,a) pairs
        x = torch.cat([state, action], dim=-1)

        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2
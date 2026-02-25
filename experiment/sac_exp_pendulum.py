"""
In the Pendulum-v1 environment, each episode has a maximum length of 200 steps, and before formal training you must use 1,000 steps of purely random actions to warm up (pre-fill) the replay buffer.
# Requirement from the materials: record the number of steps needed to reach a return of -150 (Convergence Speed)
# Requirement from the materials: evaluation must use deterministic=True

This project defines the following four core **Evaluation Metrics** to compare different algorithms:
1. Sample Efficiency: evaluate by plotting average episodic return as a function of the total number of environment interactions.
2. Convergence Speed: record how many environment interaction steps are required for the return to stably exceed the threshold of -150.
3. Asymptotic Performance: compute the average return over the last 10 evaluation episodes before training ends.
4. Model Accuracy / MSE: used to monitor model quality and guard against potential error accumulation. The computation differs by algorithm:
◦ For MBPO: compute the mean squared error (MSE) between the model-predicted next state and the true next state in observation space.
◦ For Dreamer: since transitions happen in latent space, report the "observation reconstruction error", i.e., the MSE between the true environment state and the state decoded from the latent representation.

In addition, for the overall evaluation setup, the materials specify that all algorithms (SAC, MBPO, simplified Dreamer-v1) must be run with 5 independent random seeds, and final performance should be reported as "mean ± standard deviation". If compute is limited, the number of random seeds is reduced to 3; in that case, the evaluation emphasis shifts from "final performance" to "early sample efficiency".
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from sac_agent import SACAgent
from buffer import ReplayBuffer

ENV_NAME = "Pendulum-v1"

# Experiment configuration (constants)
EPISODE_MAX_STEPS = 200 # truncate the episode at 200 steps
TRAIN_MAX_STEPS = 100_000 # maximum number of steps to train
EVAL_FREQUENCY_STEPS = 2_000 # evaluate every 2000 steps
REPLAY_PREFILL_STEPS = 1_000 # warm up for 1000 steps
BATCH_SIZE = 256 # batch size for training

CONVERGENCE_THRESHOLD = -150.0 # convergence speed threshold

ASYMPTOTIC_WINDOW_EVALS = 10 # last 10 evaluation episodes to compute asymptotic performance
EVAL_EPISODES_DEFAULT = 10 # default number of evaluation episodes
EVAL_EPISODES_DURING_TRAIN = 10 # larger number of episodes during training means more stable evaluation

SEEDS_FULL = [1, 2, 3, 4, 5] # full set of seeds
SEEDS_COMPUTE_LIMITED = [1, 2, 3] # compute-limited set of seeds

# Plotting (constants)
PLOT_FIGSIZE = (10, 6) # figure size
PLOT_STD_ALPHA = 0.2 # standard deviation alpha

def evaluate_policy(agent, env_name=ENV_NAME, eval_episodes=EVAL_EPISODES_DEFAULT, seed=None):
    """
    Requirement from the materials: during evaluation, the agent must drop stochastic exploration and deterministically output the mean of the policy.
    Run evaluation episodes using a deterministic policy.

    Output: Average episodic return
    """
    env = gym.make(env_name)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        done = False
        while not done:
            # Requirement from the materials: evaluation must use deterministic actions
            action = agent.select_action(state, deterministic=True) 
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
        
    avg_reward /= eval_episodes
    return avg_reward

class ExperimentTracker:
    """
    Requirement from the materials: run experiments with 5 independent random seeds and report results as "mean ± standard deviation".
    Also compute sample efficiency, convergence speed (>-150), asymptotic performance (mean over the last 10 evaluation episodes), and model accuracy (MSE).
    """
    def __init__(self, seeds=None):
        if seeds is None:
            seeds = SEEDS_COMPUTE_LIMITED
        self.seeds = seeds # Requirement from the materials: 5 independent random seeds (use 3 when compute is limited)
        self.results = {seed: {'steps': [], 'returns': [], 'convergence_step': None, 'model_mse': []} for seed in seeds}
        self.threshold = CONVERGENCE_THRESHOLD # Requirement from the materials: convergence-speed threshold

    def run_experiment(self, agent_class):
        """
        Outer loop: iterate over all random seeds
        """
        for seed in self.seeds:
            print(f"=== Starting run with seed: {seed} ===")
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Initialize environment and agent
            env = gym.make(ENV_NAME)
            state_dim = env.observation_space.shape[0]  # Fixed a bug when retrieving an integer dimension
            action_dim = env.action_space.shape[0]
            
            agent = agent_class(state_dim, action_dim) 
            replay_buffer = ReplayBuffer(state_dim, action_dim)
            
            # --- Core training variables ---
            max_steps = TRAIN_MAX_STEPS
            eval_freq = EVAL_FREQUENCY_STEPS  # Evaluate every N steps
            random_steps = REPLAY_PREFILL_STEPS # Requirement from the materials: warm up for 1,000 steps
            
            state, _ = env.reset(seed=seed)
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps):
                # 1. Interaction and training logic
                # Phase 1: warm-up phase (pure random exploration)
                if step < random_steps:
                    action = env.action_space.sample()
                # Phase 2: actual training and sampling phase
                else:
                    action = agent.select_action(state, deterministic=False)
                    
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store transition in the buffer (fixed argument order to match buffer.py)
                replay_buffer.add(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Start updating networks after warm-up
                if step >= random_steps:
                    agent.train(replay_buffer, batch_size=BATCH_SIZE)
                    
                if done or episode_steps >= EPISODE_MAX_STEPS: # Requirement from the materials: episodes are truncated at 200 steps max
                    state, _ = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                
                # 2. Evaluation logic (Sample Efficiency & Convergence Speed)
                if step % eval_freq == 0 and step >= random_steps:
                    eval_return = evaluate_policy(agent, ENV_NAME, eval_episodes=EVAL_EPISODES_DURING_TRAIN)
                    self.results[seed]['steps'].append(step)
                    self.results[seed]['returns'].append(eval_return)
                    print(f"Step: {step}, Eval Return: {eval_return:.2f}")
                    
                    # Check whether the convergence criterion is met (Convergence Speed)
                    if eval_return > self.threshold and self.results[seed]['convergence_step'] is None:
                        self.results[seed]['convergence_step'] = step
                        print(
                            f"*** Seed {seed} reached the convergence threshold "
                            f"({self.threshold:g}) at step {step} ***"
                        )
                        
    def report_asymptotic_performance(self):
        """
        Compute and report asymptotic performance: average return over the last 10 evaluation episodes before training ends (mean ± std).
        """
        final_returns = []
        for seed in self.seeds:
            # Get the mean return of the last N evaluations for this seed
            if len(self.results[seed]['returns']) >= ASYMPTOTIC_WINDOW_EVALS:
                last_n_returns = np.mean(self.results[seed]['returns'][-ASYMPTOTIC_WINDOW_EVALS:])
            else:
                last_n_returns = np.mean(self.results[seed]['returns'])
            final_returns.append(last_n_returns)
            
        mean_perf = np.mean(final_returns)
        std_perf = np.std(final_returns)
        print(
            f"\n[Evaluation Results] Asymptotic Performance "
            f"(last {ASYMPTOTIC_WINDOW_EVALS} evals): {mean_perf:.2f} ± {std_perf:.2f}"
        )

    def plot_sample_efficiency(self):
        """
        Plot the sample-efficiency curve: average episodic return vs total environment interactions.
        """
        all_returns = np.array([self.results[seed]['returns'] for seed in self.seeds])
        steps = self.results[self.seeds[0]]['steps']  # Fixed a bug where a list was used as a dict key
        
        mean_returns = np.mean(all_returns, axis=0)
        std_returns = np.std(all_returns, axis=0)
        
        plt.figure(figsize=PLOT_FIGSIZE)
        plt.plot(steps, mean_returns, label="SAC (Mean)")
        plt.fill_between(steps, mean_returns - std_returns, mean_returns + std_returns, alpha=PLOT_STD_ALPHA)
        plt.axhline(self.threshold, color='r', linestyle='--', label=f"Convergence Threshold ({self.threshold:g})")
        plt.xlabel("Total Environment Interactions")
        plt.ylabel("Average Episodic Return")
        plt.title(f"Sample Efficiency on {ENV_NAME}")
        plt.legend()
        plt.show()

def main():
    # Requirement from the materials: initialize 3 seeds for the compute-limited setting
    tracker = ExperimentTracker(seeds=SEEDS_COMPUTE_LIMITED)
    tracker.run_experiment(SACAgent)
    tracker.report_asymptotic_performance()
    tracker.plot_sample_efficiency()

if __name__ == "__main__":
    main()
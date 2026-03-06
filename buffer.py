"""
A simple FIFO experience replay buffer for Off-policy RL algorithms (e.g., SAC, DDPG, TD3).
Stores transitions (s, a, r, s', done) and efficiently samples mini-batches.
max_size: the maximum number of experiences to store in the buffer.

How to use:
    # 1. Initialize the buffer (e.g., Pendulum-v1 has state_dim=3, action_dim=1)
    buffer = ReplayBuffer(state_dim=3, action_dim=1, max_size=1000000)

    # 2. Add an experience tuple during environment interaction
    # Note: 'done' should be a float (0.0 for False, 1.0 for True)
    buffer.add(state, action, reward, next_state, done)

    # 3. Sample a batch of experiences during network update
    # Returns PyTorch Tensors automatically pushed to the correct device (CPU/GPU)
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=256)
============================================================
Update:
add sample_sequences method to sample contiguous sequences for sequence models (e.g., Dreamer world model).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch


class ReplayBuffer:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = int(1e6),
        *,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.max_size = max_size
        self.ptr = 0  # Pointer to the current empty spot
        self.size = 0 # Current number of elements in the buffer

        # Using numpy arrays for extremely fast memory allocation and writing
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        # done: for learning/bootstrapping (typically "terminated", not time-limit truncation)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        # episode_end: true episode boundary for sequence models / RNNs (terminated OR truncated)
        self.episode_end = np.zeros((max_size, 1), dtype=np.float32)
        
        # Determine device so the buffer can automatically push batches to the desired device.
        # If `device` is explicitly provided, respect it (important for reproducibility and avoiding extra copies).
        if device is not None:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

    def add(self, state, action, reward, next_state, done, episode_end=None):
        """
        Add a new piece of experience to the buffer.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.episode_end[self.ptr] = float(done if episode_end is None else episode_end)

        # Advance pointer, wrap around if it hits max_size (FIFO mechanism)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, states, actions, rewards, next_states, dones, episode_ends=None):
        """
        Add a batch of experiences to the buffer efficiently.
        All inputs should be numpy arrays of shape (batch_size, ...).
        """
        batch_size = states.shape[0]
        if episode_ends is None:
            episode_ends = dones

        # If the batch is larger than the buffer, only keep the latest max_size elements
        if batch_size >= self.max_size:
            self.state[:] = states[-self.max_size:]
            self.action[:] = actions[-self.max_size:]
            self.reward[:] = rewards[-self.max_size:]
            self.next_state[:] = next_states[-self.max_size:]
            self.done[:] = dones[-self.max_size:]
            self.episode_end[:] = episode_ends[-self.max_size:]
            self.ptr = 0
            self.size = self.max_size
            return

        # Calculate how many elements we can fit before wrapping around
        space_left = self.max_size - self.ptr
        
        if batch_size <= space_left:
            # Fits without wrapping
            self.state[self.ptr : self.ptr + batch_size] = states
            self.action[self.ptr : self.ptr + batch_size] = actions
            self.reward[self.ptr : self.ptr + batch_size] = rewards
            self.next_state[self.ptr : self.ptr + batch_size] = next_states
            self.done[self.ptr : self.ptr + batch_size] = dones
            self.episode_end[self.ptr : self.ptr + batch_size] = episode_ends
            self.ptr = (self.ptr + batch_size) % self.max_size
        else:
            # Wraps around
            self.state[self.ptr : self.max_size] = states[:space_left]
            self.action[self.ptr : self.max_size] = actions[:space_left]
            self.reward[self.ptr : self.max_size] = rewards[:space_left]
            self.next_state[self.ptr : self.max_size] = next_states[:space_left]
            self.done[self.ptr : self.max_size] = dones[:space_left]
            self.episode_end[self.ptr : self.max_size] = episode_ends[:space_left]
            
            remain = batch_size - space_left
            self.state[:remain] = states[space_left:]
            self.action[:remain] = actions[space_left:]
            self.reward[:remain] = rewards[space_left:]
            self.next_state[:remain] = next_states[space_left:]
            self.done[:remain] = dones[space_left:]
            self.episode_end[:remain] = episode_ends[space_left:]
            self.ptr = remain
            
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):
        """
        Sample a random batch of experiences and convert them to PyTorch Tensors.
        """
        # Randomly choose 'batch_size' indices from the currently filled portion
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

    def sample_recent_states(self, batch_size: int, recent_frac: float = 0.25):
        """
        Sample states from the most recent fraction of the replay buffer.

        Useful for MBPO start-state sampling where recent data is usually closer
        to the current policy/state distribution.
        """
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        frac = float(np.clip(recent_frac, 1e-6, 1.0))
        recent_n = max(1, int(np.ceil(self.size * frac)))

        # Build index pool for the latest `recent_n` inserted elements.
        if self.size < self.max_size:
            # Buffer not wrapped yet: valid range is [0, size).
            start = max(0, self.size - recent_n)
            pool = np.arange(start, self.size, dtype=np.int64)
        else:
            # Buffer wrapped: latest element is at (ptr-1), oldest at ptr.
            start = (self.ptr - recent_n) % self.max_size
            if start < self.ptr:
                pool = np.arange(start, self.ptr, dtype=np.int64)
            elif start > self.ptr:
                pool = np.concatenate(
                    [
                        np.arange(start, self.max_size, dtype=np.int64),
                        np.arange(0, self.ptr, dtype=np.int64),
                    ],
                    axis=0,
                )
            else:
                # recent_n == max_size => entire buffer
                pool = np.arange(0, self.max_size, dtype=np.int64)

        ind = np.random.choice(pool, size=batch_size, replace=pool.size < batch_size)
        return torch.FloatTensor(self.state[ind]).to(self.device)

    def sample_with_episode_end(self, batch_size: int):
        """
        Sample a random batch and also return `episode_end` (terminated OR truncated).

        Returns tensors:
          states:       (B, state_dim)
          actions:      (B, action_dim)
          rewards:      (B, 1)
          next_states:  (B, state_dim)
          dones:        (B, 1)  # done_for_learning (typically terminated)
          episode_ends: (B, 1)  # true episode boundary (terminated OR truncated)
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device),
            torch.FloatTensor(self.episode_end[ind]).to(self.device),
        )

    def sample_sequences(self, batch_size: int, seq_len: int, *, avoid_episode_crossing: bool = True):
        """
        Sample contiguous sequences for sequence models (e.g., Dreamer world model).

        Returns tensors:
          states:      (B, T, state_dim)
          actions:     (B, T, action_dim)
          rewards:     (B, T, 1)
          next_states: (B, T, state_dim)
          dones:       (B, T, 1)  # learning done (typically terminated)
          episode_ends:(B, T, 1)  # true episode boundary (terminated OR truncated)
        """
        if self.size < seq_len + 1:
            raise ValueError(f"Not enough data to sample sequences: size={self.size}, seq_len={seq_len}")

        max_start = self.size - seq_len

        if avoid_episode_crossing:
            # We disallow sequences that cross an episode boundary.
            #
            # In this buffer `episode_end[t]=1` indicates a true episode boundary at index t
            # (either terminated OR truncated). Crossing the boundary would mix transitions
            # from different episodes in a single sequence.
            #
            # Condition: for a start index s, require dones[s : s+seq_len-1] are all zeros.
            # (The last element can be terminal; it only affects the next step outside the sampled window.)
            end_flat = self.episode_end[: self.size, 0].astype(np.int32)  # (size,)
            window = seq_len - 1
            if window <= 0:
                valid_starts = np.arange(max_start, dtype=np.int64)
            else:
                csum = np.concatenate([[0], np.cumsum(end_flat, dtype=np.int64)])
                # end_count[s] = sum(end_flat[s : s+window])
                end_count = csum[window:] - csum[:-window]  # (size-window+1,)
                valid_starts = np.where(end_count[:max_start] == 0)[0]

            if valid_starts.size == 0:
                raise ValueError(
                    "No valid episode-contained sequences found. "
                    "Try decreasing seq_len, increasing replay_prefill_steps, or collecting longer episodes."
                )
            starts = np.random.choice(valid_starts, size=batch_size, replace=valid_starts.size < batch_size)
        else:
            starts = np.random.randint(0, max_start, size=batch_size)

        idx = starts[:, None] + np.arange(seq_len)[None, :]

        return (
            torch.FloatTensor(self.state[idx]).to(self.device),
            torch.FloatTensor(self.action[idx]).to(self.device),
            torch.FloatTensor(self.reward[idx]).to(self.device),
            torch.FloatTensor(self.next_state[idx]).to(self.device),
            torch.FloatTensor(self.done[idx]).to(self.device),
            torch.FloatTensor(self.episode_end[idx]).to(self.device),
        )
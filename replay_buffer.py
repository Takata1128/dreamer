from typing import Tuple
import torch
import numpy as np


class TransitionBuffer:
    def __init__(
        self,
        capacity,
        obs_shape: Tuple[int],
        action_size: int,
        obs_type=np.float32,
        action_type=np.float32,
    ):

        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.obs_type = obs_type
        self.action_type = action_type
        self.idx = 0
        self.full = False
        self.observation = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.action = np.empty((capacity, action_size), dtype=np.float32)
        self.reward = np.empty((capacity,), dtype=np.float32)
        self.terminal = np.empty((capacity,), dtype=bool)

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ):
        self.observation[self.idx] = obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.terminal[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _sample_idx(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.capacity if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.capacity
            valid_idx = not self.idx in idxs[1:]
        return idxs

    def _retrieve_batch(self, idxs, n, l):
        vec_idxs = idxs.transpose().reshape(-1)
        observation = self.observation[vec_idxs]
        return (
            observation.reshape(l, n, *self.obs_shape),
            self.action[vec_idxs].reshape(l, n, -1),
            self.reward[vec_idxs].reshape(l, n),
            self.terminal[vec_idxs].reshape(l, n),
        )

    def sample(self, batch_size, chunk_length):
        chunk_length += 1
        obs, act, rew, term = self._retrieve_batch(
            np.asarray([self._sample_idx(chunk_length) for _ in range(batch_size)]),
            batch_size,
            chunk_length,
        )
        obs, act, rew, term = self._shift_sequences(obs, act, rew, term)

        return obs, act, rew, term

    def _shift_sequences(self, obs, actions, rewards, terminals):
        obs = obs[1:]
        actions = actions[:-1]
        rewards = rewards[:-1]
        terminals = terminals[:-1]
        return obs, actions, rewards, terminals


class ReplayBuffer(object):
    """
    RNNを用いたReplayBuffer
    """

    def __init__(
        self, capacity, observation_shape, action_dim, obs_dtype, action_dtype
    ):
        self.capacity = capacity

        self.observations = np.zeros((capacity, *observation_shape), dtype=obs_dtype)
        self.actions = np.zeros((capacity, action_dim), dtype=action_dtype)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.bool)

        self.index = 0
        self.is_filled = False

    def push(self, observation, action, reward, done):
        """
        リプレイバッファに経験を追加
        """
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        """
        経験をリプレイバッファからサンプル
        """
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []

        for _ in range(batch_size):
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(
                    initial_index <= episode_borders, episode_borders < final_index
                ).any()
            sampled_indexes += list(range(initial_index, final_index + 1))

        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, *self.observations.shape[1:]
        )
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions_shape[1]
        )
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1
        )
        sampled_done = self.done[sampled_indexes].reshape(batch_size, chunk_length, 1)

        return sampled_observations, sampled_actions, sampled_rewards, sampled_done

    def __len__(self):
        return self.capacity if self.is_filled else self.index

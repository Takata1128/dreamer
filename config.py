from typing import Tuple
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class MinAtarConfig:
    # env desc
    env: str
    obs_shape: Tuple
    action_size: int
    action_repeat: int = 1

    # buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    # training desc
    train_episodes: int = 100
    train_steps: int = int(5e6)
    train_every: int = 50
    collect_intervals: int = 5
    batch_size: int = 50
    chunk_length: int = 50

    # latent_space desc
    embedding_size: int = 200
    rssm_node_size: int = 200
    rnn_hidden_size: int = 200
    state_dim: int = 20

    # objective desc
    clip_grad_norm: float = 100.0
    gamma: float = 0.99
    lambda_: float = 0.99
    horizon: int = 10
    model_lr: float = 2e-4
    actor_lr: float = 4e-5
    critic_lr: float = 1e-4
    kl_balance_scale: float = 0.8
    free_nats: float = 0.0

    # actor critic
    actor_entropy_scale: float = 1e-3

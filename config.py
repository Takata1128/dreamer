from typing import Tuple
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class MinAtarConfig:
    """
    Trainingの設定
    """

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
    seed_episodes: int = 1000
    train_steps: int = int(5e6)
    train_every: int = 50
    collect_intervals: int = 5
    batch_size: int = 50
    chunk_length: int = 50
    save_every: int = int(5e4)
    model_dir: str = "results"

    # latent_space desc
    embedding_size: int = 200
    rssm_node_size: int = 200
    rnn_hidden_dim: int = 200
    class_size: int = 20
    category_size: int = 20
    state_dim: int = category_size * class_size

    # objective desc
    clip_grad_norm: float = 100.0
    gamma: float = 0.99
    lambda_: float = 0.95
    horizon: int = 10
    model_lr: float = 2e-4
    actor_lr: float = 4e-5
    critic_lr: float = 1e-4
    eps: float = 1e-4
    kl_loss_scale: float = 0.1
    reward_loss_scale: float = 1.0
    discount_loss_scale: float = 5.0
    kl_balance_scale: float = 0.8
    free_nats: float = 0.0
    slow_target_update: int = 100

    # actor critic desc
    actor_entropy_scale: float = 1e-3

    # exploration desc
    train_noise: float = 0.4
    expl_min: float = 0.05
    expl_decay: float = 7000.0

    # eval desc
    eval_episode: int = 50

from typing import Tuple
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class DQNMinAtarConfig:
    """
    Trainingの設定
    """

    # env desc
    env: str
    id: int

    # training desc
    batch_size: int = 32
    n_steps: int = int(5e6)
    train_every: int = 50
    save_every: int = int(5e4)
    model_dir: str = "results"

    # replay buffer desc
    buffer_size: int = int(1e6)
    seed_buffer_size: int = int(1e4)

    # Q network desc
    qnet_lr: float = 1e-4
    target_update_interval: int = 2000

    # hyper parameter desc
    beta_begin: float = 0.4
    beta_end: float = 1.0
    beta_decay: int = 5000000

    epsilon_begin: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 500000

    gamma: float = 0.99

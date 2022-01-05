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
    pixel: bool = True
    action_repeat: int = 1

    # buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    # training desc
    train_steps: int = int(5e6)
    train_every: int = 50
    collect_intervals: int = 5
    batch_size: int = 50
    seq_len: int = 50

    # latent_space desc
    embedding_size: int = 200
    hidden_size: int = 200

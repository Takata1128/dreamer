import torch
import gym
import time
import os
import argparse
import numpy as np
from config import MinAtarConfig
from trainer import Trainer

from wrapper import GymMinAtar, OneHotAction


seed = 1


def main(args):
    env_name = args.env_name
    exp_id = args.id

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")

    env = OneHotAction(GymMinAtar(env_name))
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    obs_dtype = bool
    action_dtype = np.float32
    batch_size = args.batch_size
    chunk_length = args.chunk_length

    config = MinAtarConfig(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype=obs_dtype,
        action_dtype=action_dtype,
        chunk_length=chunk_length,
        batch_size=batch_size,
    )

    trainer = Trainer(config, device)

    """
    training loop
    """

    trainer.collect_seed_episodes()

    for episode in range(config.train_episodes):
        start = time.time()
        obs, done = env.reset(), False
        total_reward = 0

        while not done:
            action = policy(obs)


if __name__ == "__main__":
    parser = argparse
    parser.add_argument("--env", type=str, help="mini atari env name")
    parser.add_argument("--id", type=str, default="0", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=1, help="Random Seed")
    parser.add_argument("--no_cuda", action="store_false", help="Use GPU")
    args = parser.parse_args()
    main(args)

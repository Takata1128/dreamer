from dqn.trainer import DQNTrainer
from dqn.qnet import QNetwork
from dqn.config import DQNMinAtarConfig
from dqn.replay_buffer import PrioritizedReplayBuffer
from wrapper import GymMinAtar, breakoutPOMDP
import argparse
import numpy as np
import torch
import gym
import os


class TorchFrame(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def observation(self, observation):
        return torch.as_tensor(observation)


def main(args):
    env_name = args.env
    exp_id = args.id

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")
    print("using :", device)

    env = GymMinAtar(env_name)

    if args.pomdp and env_name == "breakout":
        print("pomdp")
        env = breakoutPOMDP(env)
    else:
        print("mdp")

    env = TorchFrame(env)

    config = DQNMinAtarConfig(
        env=env_name,
        id=exp_id,
    )

    result_dir = os.path.join(config.model_dir, "{}_{}".format(env_name, exp_id))
    model_dir = os.path.join(result_dir, "dqn_models")
    os.makedirs(model_dir, exist_ok=True)

    trainer = DQNTrainer(env, config, device)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="breakout", help="mini atari env name"
    )
    parser.add_argument(
        "--id", type=str, default="pomdp_dqn_sparse", help="Experiment ID"
    )
    parser.add_argument("--pomdp", action="store_false", help="Use pomdp environment")
    parser.add_argument("--seed", type=int, default=1234786, help="Random Seed")
    parser.add_argument("--no_cuda", action="store_true", help="Use GPU")
    args = parser.parse_args()
    main(args)

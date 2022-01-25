import torch
import gym
import os
import argparse
import numpy as np
from config import MinAtarConfig
from wrapper import GymMinAtar, OneHotAction, breakoutPOMDP

from evaluator import DreamerV2Evaluator, DQNEvaluator
import matplotlib.pyplot as plt


class TorchFrame(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def observation(self, observation):
        return torch.as_tensor(observation)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")
    print("using :", device)

    env = OneHotAction(GymMinAtar(args.env))
    dqn_env = TorchFrame(GymMinAtar(args.env))

    if args.pomdp and args.env == "breakout":
        print("pomdp")
        env = breakoutPOMDP(env)
        dqn_env = TorchFrame(breakoutPOMDP(dqn_env))
    else:
        print("mdp")

    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    obs_dtype = bool
    action_dtype = np.float32
    config = MinAtarConfig(
        env=args.env,
        id=args.drm_id,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype=obs_dtype,
        action_dtype=action_dtype,
    )  # dreamer の configを共有

    fig, ax = plt.subplots()

    if args.dqn_id is not None and args.dqn_id != "":
        dqn_result_dir = os.path.join(
            config.model_dir, "{}_{}".format(args.env, args.dqn_id)
        )
        dqn_models_dir = os.path.join(
            dqn_result_dir, "dqn_models"
        )  # dir to save learnt models

        dqn_evaluator = DQNEvaluator(dqn_env, config, device)

        dqn_indices, dqn_scores = dqn_evaluator.evalueate_sequence(
            dqn_result_dir, dqn_models_dir
        )
        dqn_evaluator.close()

        ax.plot(dqn_indices, dqn_scores, label="DDQN")

    if args.drm_id is not None and args.drm_id != "":
        drm_result_dir = os.path.join(
            config.model_dir, "{}_{}".format(args.env, args.drm_id)
        )
        drm_models_dir = os.path.join(
            drm_result_dir, "models"
        )  # dir to save learnt models
        dreamer_evaluator = DreamerV2Evaluator(env, config, device)
        dreamer_indices, dreamer_scores = dreamer_evaluator.evalueate_sequence(
            drm_result_dir, drm_models_dir
        )

        dreamer_evaluator.close()
        ax.plot(dreamer_indices, dreamer_scores, label="DreamerV2")

    plt.savefig("pomdp_eval.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="breakout", help="mini atari env name"
    )
    parser.add_argument("--pomdp", action="store_false", help="Use pomdp environment")
    parser.add_argument(
        "--drm_id", type=str, default="pomdp", help="Dreamerv2 Experiment ID"
    )
    parser.add_argument(
        "--dqn_id", type=str, default="pomdp_dqn_sparse", help="DQN Experiment ID"
    )

    parser.add_argument("--seed", type=int, default=1, help="Random Seed")
    parser.add_argument("--no_cuda", action="store_true", help="Use GPU")
    args = parser.parse_args()
    main(args)

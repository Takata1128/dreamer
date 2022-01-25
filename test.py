import torch
import gym
import os
import argparse
import numpy as np
from config import MinAtarConfig
from wrapper import GymMinAtar, OneHotAction, breakoutPOMDP

from evaluator import DreamerV2Evaluator, DQNEvaluator
import matplotlib.pyplot as plt


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

    if args.pomdp and args.env == "breakout":
        print("pomdp")
        env = breakoutPOMDP(env)
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
    )

    drm_result_dir = os.path.join(
        config.model_dir, "{}_{}".format(args.env, args.drm_id)
    )
    drm_models_dir = os.path.join(drm_result_dir, "models")  # dir to save learnt models
    dreamer_evaluator = DreamerV2Evaluator(env, config, device)
    images, predicted_images = dreamer_evaluator.view_episode(
        os.path.join(drm_models_dir, "models_best.pth")
    )
    images[0].save(
        args.env + ".gif", save_all=True, append_images=images[1:], duration=50
    )
    predicted_images[0].save(
        args.env + "_predicted.gif",
        save_all=True,
        append_images=predicted_images[1:],
        duration=50,
    )
    dreamer_evaluator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="breakout", help="mini atari env name"
    )
    parser.add_argument("--pomdp", action="store_false", help="Use pomdp environment")
    parser.add_argument(
        "--drm_id", type=str, default="pomdp", help="Dreamerv2 Experiment ID"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random Seed")
    parser.add_argument("--no_cuda", action="store_true", help="Use GPU")
    args = parser.parse_args()
    main(args)

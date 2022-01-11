import torch
import gym
import time
import os
import argparse
import wandb
import numpy as np
from config import MinAtarConfig
from trainer import Trainer
from agent import Agent
from wrapper import GymMinAtar, OneHotAction

from models.rssm import RecurrentStateSpaceModel
from models.observation_model import ObsEncoder, ObsDecoder
from models.action_model import ActionModel

import matplotlib.pyplot as plt


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

    env = OneHotAction(GymMinAtar(env_name))
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    obs_dtype = bool
    action_dtype = np.float32
    config = MinAtarConfig(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype=obs_dtype,
        action_dtype=action_dtype,
    )

    result_dir = os.path.join(config.model_dir, "{}_{}".format(env_name, exp_id))
    model_dir = os.path.join(result_dir, "models")  # dir to save learnt models
    os.makedirs(model_dir, exist_ok=True)

    evaluator = Evaluator(config, device)

    eval_scores = []
    for f in sorted(os.listdir(model_dir)):
        if f == "models_best.pth":
            continue
        eval_score = evaluator.evaluate(env, os.path.join(model_dir, f))
        eval_scores.append(eval_score)
    fig, ax = plt.subplots()
    ax.plot(eval_scores)
    plt.savefig(os.path.join(result_dir, "eval.png"))


class Evaluator(object):
    """
    モデルの評価を行うためのクラス
    """

    def __init__(self, config, device):
        self.device = device
        self.config = config
        self.action_size = config.action_size

    def load_model(self, model_path):
        """
        保存していたモデルをロード
        """
        saved_dict = torch.load(model_path)
        obs_shape = self.config.obs_shape
        action_size = self.config.action_size
        rnn_hidden_dim = self.config.rnn_hidden_dim
        category_size = self.config.category_size
        class_size = self.config.class_size
        state_dim = category_size * class_size

        embedding_size = self.config.embedding_size
        rssm_node_size = self.config.rssm_node_size

        self.rssm = (
            RecurrentStateSpaceModel(
                rssm_node_size,
                embedding_size,
                state_dim,
                action_size,
                rnn_hidden_dim,
                category_size,
                class_size,
            )
            .to(self.device)
            .eval()
        )
        self.action_model = (
            ActionModel(
                state_dim=state_dim,
                rnn_hidden_dim=rnn_hidden_dim,
                action_dim=action_size,
            )
            .to(self.device)
            .eval()
        )

        self.encoder = ObsEncoder(obs_shape, embedding_size).to(self.device).eval()
        self.decoder = (
            ObsDecoder(obs_shape, state_dim + rnn_hidden_dim).to(self.device).eval()
        )

        self.rssm.load_state_dict(saved_dict["RSSM"], strict=False)
        self.encoder.load_state_dict(saved_dict["ObsEncoder"], strict=False)
        self.decoder.load_state_dict(saved_dict["ObsDecoder"], strict=False)
        self.action_model.load_state_dict(saved_dict["ActionModel"], strict=False)

    def evaluate(self, env, model_path):
        """
        評価するエピソード分プレイさせてモデルを評価
        """
        self.load_model(model_path)
        eval_episodes = self.config.eval_episode
        agent = Agent(self.encoder, self.rssm, self.action_model, self.config)
        eval_scores = []
        for e in range(eval_episodes):
            obs, done = env.reset(), False
            score = 0
            while not done:
                action, _ = agent(obs, not done)
            next_obs, rew, done, _ = env.step(action.squeeze().cpu().numpy())
            score += rew
            obs = next_obs
            eval_scores.append(score)
        print(
            "average evaluation score for model at "
            + model_path
            + " = "
            + str(np.mean(eval_scores))
        )
        env.close()
        return np.mean(eval_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="breakout", help="mini atari env name"
    )
    parser.add_argument("--id", type=str, default="0", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=1, help="Random Seed")
    parser.add_argument("--no_cuda", action="store_true", help="Use GPU")
    args = parser.parse_args()
    main(args)

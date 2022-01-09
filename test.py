import torch
import gym
import time
import os
import argparse
import numpy as np
from config import MinAtarConfig
from trainer import Trainer

from wrapper import GymMinAtar, OneHotAction


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

    trainer = Trainer(config, device)

    """
    training loop
    """

    # ランダム方策でバッファに経験をためる
    trainer.collect_seed_episodes(env)

    start = time.time()
    obs, done = env.reset(), False
    total_reward = 0

    # 学習ループ
    for step in range(1, config.train_steps + 1):
        # NNのパラメータ更新
        trainer.train_batch()

        with torch.no_grad():
            embed = trainer.encoder(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(trainer.device)
            )
            prior_logit, posterior_logit, rnn_hidden = trainer.rssm(
                state, action, rnn_hidden, embed
            )
            state = trainer.rssm.get_stoch_state(posterior_logit)
            action, action_dist = trainer.action_model(state, rnn_hidden)
            action = trainer.action_model.add_exploration(action, iter).detach()
            action_ent = torch.mean(action_dist.entropy()).item()
            episode_actor_ent.append(action_ent)

        next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
        score += rew

        trainer.replay_buffer.push(obs, action.squeeze(0).cpu().numpy(), rew, done)
        if done:
            obs, score = env.reset(), 0
            done = False
            prev_state = torch.zeros(1, trainer.state_dim).to(trainer.device)
            prev_action = torch.zero(1, trainer.action_size).to(trainer.device)
            episode_actor_ent = []
        else:
            obs = next_obs
            prev_state = state
            prev_action = action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="breakout", help="mini atari env name"
    )
    parser.add_argument("--id", type=str, default="0", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=1, help="Random Seed")
    parser.add_argument("--no_cuda", action="store_false", help="Use GPU")
    args = parser.parse_args()
    main(args)

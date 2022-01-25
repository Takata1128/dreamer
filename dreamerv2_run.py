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
from wrapper import GymMinAtar, OneHotAction, breakoutPOMDP

"""
学習ループ
"""


def main(args):
    wandb.login()
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

    if args.pomdp and env_name == "breakout":
        print("pomdp")
        env = breakoutPOMDP(env)
    else:
        print("mdp")

    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    obs_dtype = bool
    action_dtype = np.float32
    config = MinAtarConfig(
        env=env_name,
        id=exp_id,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype=obs_dtype,
        action_dtype=action_dtype,
    )

    result_dir = os.path.join(config.model_dir, "{}_{}".format(env_name, exp_id))
    model_dir = os.path.join(result_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    config_dict = config.__dict__
    trainer = Trainer(config, device)
    agent = Agent(trainer.encoder, trainer.rssm, trainer.action_model, config)

    with wandb.init(project="My Project in DL Autumn Seminar", config=config_dict):
        print("training start!")
        start_t = time.time()
        train_metrics = {}

        # ランダム方策でバッファに経験をためる
        trainer.collect_seed_episodes(env)

        obs, done = env.reset(), False
        total_reward = 0
        episode_count = 0
        scores = []
        episode_actor_ent = []
        best_mean_score = 0
        best_save_path = os.path.join(model_dir, "models_best.pth")

        # 学習ループ
        for step in range(1, config.train_steps + 1):
            # NNのパラメータ更新
            if step % trainer.config.train_every == 0:
                train_metrics = trainer.train_batch(train_metrics)
            # valueネットワークの同期
            if step % trainer.config.slow_target_update == 0:
                trainer.update_target()
            # モデルの保存
            if step % trainer.config.save_every == 0:
                trainer.save_model(step)

            # 1ステップ行動
            with torch.no_grad():
                action, action_dist = agent(obs, not done, step)
                action_ent = torch.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)

            next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            total_reward += rew

            # エピソード終了判定
            if done:
                trainer.replay_buffer.push(
                    obs, action.squeeze(0).cpu().numpy(), rew, done
                )
                train_metrics["train_rewards"] = total_reward
                train_metrics["action_ent"] = np.mean(episode_actor_ent)
                wandb.log(train_metrics, step=step)
                scores.append(total_reward)
                if len(scores) > 100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    if current_average > best_mean_score:
                        best_t = time.time()
                        elapsed = best_t - start_t
                        best_mean_score = current_average
                        print(
                            "time : {} ,step : {} ,episode : {} ,saving best model with mean score : {}".format(
                                int(elapsed), step, episode_count, best_mean_score
                            ),
                        )
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)
                obs, total_reward = env.reset(), 0
                done = False
                agent.reset()  # rssmの内部状態をリセット
                episode_actor_ent = []
                episode_count += 1
            else:
                trainer.replay_buffer.push(
                    obs, action.squeeze(0).detach().cpu().numpy(), rew, done
                )
                obs = next_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="breakout", help="mini atari env name"
    )
    parser.add_argument("--pomdp", action="store_false", help="Use pomdp environment")
    parser.add_argument("--id", type=str, default="pomdp", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=1, help="Random Seed")
    parser.add_argument("--no_cuda", action="store_true", help="Use GPU")
    args = parser.parse_args()
    main(args)

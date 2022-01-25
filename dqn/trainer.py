from sentry_sdk import start_transaction
import torch
import numpy as np
from dqn.replay_buffer import PrioritizedReplayBuffer
from dqn.qnet import QNetwork
import os
import time
import wandb


class DQNTrainer:
    def __init__(self, env, config, device):
        self.env = env
        self.config = config
        self.device = device

        self.train_every = config.train_every
        self.save_every = config.save_every

        ### リプレイバッファ
        self.buffer_size = config.buffer_size
        self.seed_buffer_size = config.seed_buffer_size
        self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size)

        ### ネットワーク
        self.net = QNetwork(
            env.observation_space.shape, n_action=env.action_space.n
        ).to(device)
        self.target_net = QNetwork(
            env.observation_space.shape, n_action=env.action_space.n
        ).to(device)
        self.target_update_interval = config.target_update_interval

        ### オプティマイザとロス関数
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.qnet_lr)
        self.loss_func = torch.nn.SmoothL1Loss(reduction="none")

        ### Prioritized Experience Replayのためのパラメータβ
        beta_begin = config.beta_begin
        beta_end = config.beta_end
        beta_decay = config.beta_decay
        # betaの値を線形に増やす
        self.beta_func = lambda step: min(
            beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay)
        )

        ### 探索のためのパラメータε
        epsilon_begin = config.epsilon_begin
        epsilon_end = config.epsilon_end
        epsilon_decay = config.epsilon_decay
        # epsilonの値を線形に減らす
        self.epsilon_func = lambda step: max(
            epsilon_end,
            epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay),
        )

        ### その他ハイパーパラメータ
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.n_steps = config.n_steps

    def train(self):
        """
        指定したステップ数学習
        """

        wandb.login()
        with wandb.init(project="MinAtar with DDQN", config=self.config.__dict__):
            print("training start!")

            obs = self.env.reset()
            done = False

            train_metrics = {}
            scores = []
            best_mean_score = 0
            episode = 0
            episode_reward = 0

            loss = 0
            q_values_mean = 0

            start_t = time.time()
            for step in range(1, self.n_steps + 1):
                action = self.net.act(
                    obs.float().to(self.device), self.epsilon_func(step)
                )

                next_obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                self.replay_buffer.push([obs, action, reward, next_obs, done])
                obs = next_obs

                # Qネットワークの更新
                if (
                    len(self.replay_buffer) > self.seed_buffer_size
                    and step % self.train_every == 0
                ):
                    loss, q_values_mean = self._update(self.beta_func(step))

                # ターゲットネットワークの同期
                if step % self.target_update_interval == 0:
                    self.target_net.load_state_dict(self.net.state_dict())

                # モデルを定期的に保存
                if step % self.save_every == 0:
                    self.save_model(step)

                if done:
                    if len(self.replay_buffer) > self.seed_buffer_size:
                        train_metrics["loss"] = loss
                        train_metrics["q_values_mean"] = q_values_mean
                    train_metrics["train_rewards"] = episode_reward
                    wandb.log(train_metrics, step=step)
                    scores.append(episode_reward)
                    if len(scores) > 100:
                        scores.pop(0)
                        current_average = np.mean(scores)
                        if current_average > best_mean_score:
                            # log and save model
                            best_t = time.time()
                            elapsed = best_t - start_t
                            best_mean_score = current_average
                            print(
                                "time : {} ,step : {} ,episode : {} ,saving best model with mean score : {}".format(
                                    int(elapsed), step, episode, best_mean_score
                                )
                            )
                            save_dict = self.net.state_dict()
                            result_dir = os.path.join(
                                self.config.model_dir,
                                "{}_{}".format(self.config.env, self.config.id),
                            )
                            model_dir = os.path.join(result_dir, "dqn_models")
                            save_path = os.path.join(model_dir, "models_best.pth")
                            torch.save(save_dict, save_path)

                    episode_reward = 0
                    episode += 1
                    obs = self.env.reset()
                    done = False

    def _update(self, beta):
        (
            obs,
            action,
            reward,
            next_obs,
            done,
            indices,
            weights,
        ) = self.replay_buffer.sample(self.batch_size, beta)
        obs, action, reward, next_obs, done, weights = (
            obs.float().to(self.device),
            action.to(self.device),
            reward.to(self.device),
            next_obs.float().to(self.device),
            done.to(self.device),
            weights.to(self.device),
        )

        # 実際に選択した行動に対応する行動価値
        q_values = self.net(obs).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN
            # 1.現在のQ関数でgreedyに行動選択
            greedy_action_next = torch.argmax(self.net(next_obs), dim=1)
            # 2.対応する価値はターゲットネットワークを参照
            q_values_next = (
                self.target_net(next_obs)
                .gather(1, greedy_action_next.unsqueeze(1))
                .squeeze(1)
            )

        # ターゲット計算
        # ゲーム終了時は次状態の価値は0
        target_q_values = reward + self.gamma * q_values_next * (1 - done)

        self.optimizer.zero_grad()
        loss = (weights * self.loss_func(q_values, target_q_values)).mean()
        loss.backward()
        self.optimizer.step()

        q_values_mean = torch.mean(q_values)

        # TD誤差からサンプルの優先度を更新
        self.replay_buffer.update_priorities(
            indices, (target_q_values - q_values).abs().detach().cpu().numpy()
        )

        return loss.item(), q_values_mean.item()

    def save_model(self, iter):
        """
        モデルの保存
        """
        save_dict = self.net.state_dict()
        result_dir = os.path.join(
            self.config.model_dir,
            "{}_{}".format(self.config.env, self.config.id),
        )
        model_dir = os.path.join(result_dir, "dqn_models")
        save_path = os.path.join(model_dir, "model_%d.pth" % iter)
        torch.save(save_dict, save_path)

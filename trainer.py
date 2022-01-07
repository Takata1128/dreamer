import torch
import numpy as np
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from replay_buffer import ReplayBuffer, TransitionBuffer
from models.rssm import RecurrentStateSpaceModel
from models.encoder import Encoder
from models.decoder import ObservationModel
from models.reward_model import RewardModel
from models.action_model import ActionModel
from models.value_model import ValueModel
from agent import Agent


class Trainer(object):
    def __init__(self, config, device):
        self.device = device
        self.config = config

        self.state_dim = config.state_dim
        self.rnn_hidden_dim = config.rnn_hidden_dim
        self.action_size = config.action_size
        self.chunk_length = config.chunk_length
        self.batch_size = config.batch_size
        self.collect_intervals = config.collect_intervals
        self.seed_episodes = config.seed_episodes
        self.lambda_ = config.lambda_
        self.horizon = config.horizon
        self.kl_balance_scale = config.kl_balance_scale
        self.kl_loss_scale = config.kl_loss_scale
        self.free_nats = config.free_nats
        self.reward_loss_scale = config.reward_loss_scale
        self.actor_entropy_scale = config.actor_entropy_scale
        self.clip_grad_norm = config.clip_grad_norm

        self._model_initialize(config)
        self._optim_initialize(config)
        self._agent_initialize()

    def collect_seed_episodes(self, env):
        """
        最初、ランダム行動によりデータを集める
        """
        s, done = env.reset(), False
        for i in range(self.seed_episodes):
            a = env.action_space.sample()
            ns, r, done, _ = env.step(a)
            self.replay_buffer.push(s, a, r, done)
            if done:
                s, done = env.reset(), False
            else:
                s = ns

    def rollout_episode(self, env):
        obs, done = env.reset(), False
        total_reward = 0
        while not done:
            action = self.agent(torch.tensor(obs, dtype=torch.float32))
            next_obs, reward, done, _ = env.step(action)
            self.replay_buffer.push(obs, action, reward, done)
            obs = next_obs
            total_reward += reward
        return total_reward

    def train_batch(self):
        for update_step in range(self.collect_intervals):

            observations, actions, rewards, terminals = self.replay_buffer.sample(
                batch_size=self.batch_size, chunk_length=self.chunk_length
            )

            # 観測の前処理&Tensorの次元調整
            observations = torch.as_tensor(
                observations, dtype=torch.float32, device=self.device
            )
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            rewards = torch.as_tensor(
                actions, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)
            non_terminals = torch.as_tensor(
                1 - terminals, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)

            # 観測を低次元ベクトルに変換
            embedded_observations = self.encoder(
                observations.reshape(-1, *self.config.obs_shape)
            ).view(self.chunk_length, self.batch_size, -1)

            # 低次元の状態表現保持のためのTensor
            states = torch.zeros(
                self.chunk_length, self.batch_size, self.state_dim, device=self.device
            )
            rnn_hiddens = torch.zeros(
                self.chunk_length,
                self.batch_size,
                self.rnn_hidden_dim,
                device=self.device,
            )

            # 状態表現をゼロ初期化
            state = torch.zeros(self.batch_size, self.state_dim, device=self.device)
            rnn_hidden = torch.zeros(
                self.batch_size, self.rnn_hidden_dim, device=self.device
            )

            # priorとposterior間の KL-Loss
            kl_loss = 0
            for l in range(self.chunk_length - 1):
                (
                    (ns_prior_logit, ns_prior),
                    (
                        ns_posterior_logit,
                        ns_posterior,
                    ),
                    rnn_hidden,
                ) = self.rssm(
                    state, actions[l], rnn_hidden, embedded_observations[l + 1]
                )
                prior_dist = self.rssm.get_dist(ns_prior_logit)
                posterior_dist = self.rssm.get_dist(ns_posterior_logit)

                state = ns_posterior
                states[l + 1] = state
                rnn_hiddens[l + 1] = rnn_hidden

                alpha = self.kl_balance_scale
                kl_lhs = torch.distributions.kl.kl_divergence(
                    self.rssm.get_dist(ns_posterior_logit.detach()), prior_dist
                )

                kl_rhs = torch.distributions.kl.kl_divergence(
                    posterior_dist, self.rssm.get_dist(ns_posterior_logit.detach())
                )
                # TODO KL誤差がfree_nats以下の時は無視

                kl_loss += (
                    alpha * kl_lhs.clamp(min=self.free_nats).mean()
                    + (1 - alpha) * kl_rhs.clamp(min=self.free_nats).mean()
                )

            kl_loss /= self.chunk_length - 1

            # states[0],rnn_hiddens[0]は捨てる（ゼロ初期化のため）
            states = states[1:]
            rnn_hiddens = rnn_hiddens[1:]

            # 観測の再構成と報酬の予測
            flatten_states = states.view(-1, self.state_dim)
            flatten_rnn_hiddens = rnn_hiddens.view(-1, self.rnn_hidden_dim)
            recon_observations = self.decoder(flatten_states, flatten_rnn_hiddens).view(
                self.chunk_length - 1, self.batch_size, 3, 64, 64
            )
            predicted_rewards = self.reward_model(
                flatten_states, flatten_rnn_hiddens
            ).view(self.chunk_length - 1, self.batch_size, 1)

            # 観測と報酬の予測誤差を計算
            obs_loss = (
                0.5
                * F.mse_loss(recon_observations, observations[1:], reduction="none")
                .mean([0, 1])
                .sum()
            )
            reward_loss = 0.5 * F.mse_loss(predicted_rewards, rewards[:-1])

            # 以上のロスから勾配更新
            model_loss = kl_loss + obs_loss + reward_loss
            self.model_optimizer.zero_grad()
            model_loss.backward()
            clip_grad_norm_(self.model_params, self.clip_grad_norm)
            self.model_optimizer.step()

            ### ActionModel,ValueModelの更新
            # 勾配の流れを一度遮断
            flatten_states = flatten_states.detach()
            flatten_rnn_hiddens = flatten_rnn_hiddens.detach()

            # 現在のモデルを用いた数ステップ先の未来の状態予測
            imaginated_states = torch.zeros(
                self.imagination_horizon + 1,
                *flatten_states.shape,
                device=flatten_states.device
            )
            imaginated_rnn_hiddens = torch.zeros(
                self.imagination_horizon + 1,
                *flatten_rnn_hiddens.shape,
                device=flatten_rnn_hiddens.device
            )

            # 初期状態はリプレイバッファからサンプルした観測データ
            imaginated_states[0] = flatten_states
            imaginated_rnn_hiddens[0] = flatten_rnn_hiddens

            # open-loopで未来の状態予測を使い、想像上の軌道を作る
            for h in range(1, self.imagination_horizon + 1):
                # ActionModelにより行動を決定（行動は）
                pass

    def actor_critic_loss(self, posterior):
        pass

    def representation_loss(self, obs, actions, rewards, nonterms):
        pass

    def _actor_loss(
        self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy
    ):
        pass

    def _value_loss(self, imag_modelstates, discount, lambda_returns):
        pass

    def _obs_loss(self, obs_dist, obs):
        pass

    def _kl_loss(self, prior, posterior):
        pass

    def reward_loss(self, reward_dist, rewards):
        pass

    def _pcont_loss(self, pcont_dist, nonterms):
        pass

    def update_target(self):
        pass

    def save_model(self, iter):
        pass

    def get_save_dict(self):
        pass

    def load_save_dict(self, saved_dict):
        pass

    def _model_initialize(self, config):
        """
        世界モデルとエージェントのモデルを用意
        """
        obs_shape = config.obs_shape
        action_size = config.action_size
        rnn_hidden_dim = config.rnn_hidden_dim
        category_size = config.category_size
        class_size = config.class_size
        state_dim = category_size * class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size

        self.replay_buffer = TransitionBuffer(
            config.capacity,
            obs_shape,
            action_size,
            config.obs_dtype,
            config.action_dtype,
        )
        self.rssm = RecurrentStateSpaceModel(
            rssm_node_size,
            embedding_size,
            state_dim,
            action_size,
            rnn_hidden_dim,
            category_size,
            class_size,
        )
        self.action_model = ActionModel(
            state_dim=state_dim, rnn_hidden_dim=rnn_hidden_dim, action_dim=action_size
        )
        self.value_model = ValueModel(
            state_dim=state_dim, rnn_hidden_dim=rnn_hidden_dim
        )
        self.reward_model = RewardModel(
            state_dim=state_dim, rnn_hidden_dim=rnn_hidden_dim
        )

        self.encoder = Encoder()
        self.decoder = ObservationModel(
            state_dim=state_dim, rnn_hidden_dim=rnn_hidden_dim
        )

    def _optim_initialize(self, config):
        """
        optimizerの用意
        """
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_model.parameters())
        )
        model_optimizer = optim.Adam(
            self.model_params, lr=config.model_lr, eps=config.eps
        )
        value_optimizer = optim.Adam(
            self.value_model.parameters(), lr=config.critic_lr, eps=config.eps
        )
        action_optimizer = optim.Adam(
            self.action_model.parameters(), lr=config.actor_lr, eps=config.eps
        )

    def _agent_initialize(self):
        self.agent = Agent(self.encoder, self.rssm, self.action_model)

    def _print_summary(self):
        pass

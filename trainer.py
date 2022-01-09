import torch
import numpy as np
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from replay_buffer import ReplayBuffer, TransitionBuffer
from models.rssm import RecurrentStateSpaceModel
from models.observation_model import ObsEncoder, ObsDecoder
from models.reward_model import RewardModel
from models.action_model import ActionModel
from models.value_model import ValueModel
from models.discount_model import DiscountModel
from agent import Agent


class Trainer(object):
    def __init__(self, config, device):
        self.device = device
        self.config = config

        self.obs_shape = config.obs_shape
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
        self.discount_loss_scale = config.discount_loss_scale
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

    def observations_reconstruct(
        self, state, rnn_hidden, embedded_obs, action, nonterms
    ):
        """
        観測データからchunk_length分の観測を再構成
        """
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
        prior_logits = []
        posterior_logits = []
        for l in range(self.chunk_length - 1):
            prev_action = action[l] * nonterms[l]
            prev_state = state * nonterms[l]
            prior_logit, posterior_logit, rnn_hidden = self.rssm(
                prev_state, prev_action, rnn_hidden, embedded_obs[l]
            )
            states[l + 1] = self.rssm.get_stoch_state(posterior_logit)
            rnn_hiddens[l + 1] = rnn_hidden
            prior_logits.append(prior_logit)
            posterior_logits.append(posterior_logit)

        prior_logits = torch.stack(prior_logits, dim=0)
        posterior_logits = torch.stack(prior_logits, dim=0)

        return prior_logits, posterior_logits, states, rnn_hiddens

    def rollout_imagination(
        self, imagination_horizon, flatten_states, flatten_rnn_hiddens
    ):
        imaginated_states = []
        imaginated_rnn_hiddens = []
        action_entropy = []
        imag_log_probs = []

        for t in range(imagination_horizon):
            action, action_dist = self.action_model(flatten_states, flatten_rnn_hiddens)
            flatten_states_prior_logits, flatten_rnn_hiddens = self.rssm.prior(
                flatten_states, action, flatten_rnn_hiddens
            )
            flatten_states = self.rssm.get_stoch_state(flatten_states_prior_logits)
            imaginated_states.append(flatten_states)
            imaginated_rnn_hiddens.append(flatten_rnn_hiddens)
            action_entropy.append(action_dist.entropy())
            imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))

        imaginated_states = torch.stack(imaginated_states, dim=0)
        imaginated_rnn_hiddens = torch.stack(imaginated_rnn_hiddens, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)

        return (
            imaginated_states,
            imaginated_rnn_hiddens,
            imag_log_probs,
            action_entropy,
        )

    def train_batch(self):
        """
        バッファからサンプルしたデータから世界モデルとエージェントを学習
        """
        actor_l = []
        value_l = []
        obs_l = []
        model_l = []
        reward_l = []
        prior_ent_l = []
        post_ent_l = []
        kl_l = []
        pcont_l = []
        mean_targ = []
        min_targ = []
        max_targ = []
        std_targ = []

        for update_step in range(self.collect_intervals):
            # バッファから経験をサンプル (chunk_length,batch_size,*)
            observations, actions, rewards, terminals = self.replay_buffer.sample(
                batch_size=self.batch_size, chunk_length=self.chunk_length
            )
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

            (
                model_loss,
                kl_loss,
                obs_loss,
                reward_loss,
                pcont_loss,
                prior_dist,
                post_dist,
                states,
                rnn_hiddens,
            ) = self.representation_loss(observations, actions, rewards, non_terminals)

            self.model_optimizer.zero_grad()
            model_loss.backward()
            model_params = (
                list(self.encoder.parameters())
                + list(self.rssm.parameters())
                + list(self.decoder.parameters())
                + list(self.reward_model.parameters())
            )
            grad_norm_model = torch.nn.utils.clip_grad_norms_(model_params)
            self.model_optimizer.step()

            actor_loss, value_loss = self.actor_critic_loss(states, rnn_hiddens)

            self.action_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            actor_loss.backward()
            value_loss.backward()

            grad_norm_actor = torch.nn.utils.clip_grad_norm_(
                self.action_model.parameters()
            )
            grad_norm_value = torch.nn.utils.clip_grad_norm_(
                self.value_model.parameters()
            )

            self.action_optimizer.step()
            self.value_optimizer.step()

    def actor_critic_loss(self, states, rnn_hiddens):
        """
        Actor Criticのロス計算
        """
        with torch.no_grad():
            flatten_states = states.view(-1, self.state_dim).detach()
            flatten_rnn_hiddens = rnn_hiddens.view(-1, self.rnn_hidden_dim).detach()

        # 世界モデルのパラメータは凍結
        (
            imaginated_states,
            imaginated_rnn_hiddens,
            imag_log_prob,
            policy_entropy,
        ) = self.rollout_imagination(self.horizon, flatten_states, flatten_rnn_hiddens)

        # ActorCritic以外のパラメータは凍結
        imag_reward_dist = self.reward_model(imaginated_states, imaginated_rnn_hiddens)
        imag_reward = imag_reward_dist.mean
        imag_value_dist = self.target_value_model(
            imaginated_states, imaginated_rnn_hiddens
        )
        imag_value = imag_value_dist.mean
        discount_dist = self.discount_model(imaginated_states, imaginated_rnn_hiddens)
        discount_arr = self.discount * torch.round(discount_dist.base_dist.probs)

        actor_loss, discount, lambda_returns = self._actor_loss(
            imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy
        )
        value_loss = self._value_loss(
            imaginated_states, imaginated_rnn_hiddens, discount, lambda_returns
        )
        return actor_loss, value_loss

    def representation_loss(self, obs, actions, rewards, nonterms):
        """
        世界モデルのロス計算
        """
        embed = self.encoder(obs)
        state = torch.zeros(self.batch_size, self.state_dim, device=self.device)
        rnn_hidden = torch.zeros(
            self.batch_size, self.rnn_hidden_dim, device=self.device
        )

        prior_logits, posterior_logits, rnn_hiddens = self.observations_reconstruct(
            state, rnn_hidden, embed, actions, nonterms
        )

        posterior = self.rssm.get_stoch_state(posterior_logits)
        flatten_posterior = posterior.view(-1, self.state_dim)
        flatten_rnn_hiddens = rnn_hiddens.view(-1, self.rnn_hidden_dim)

        obs_dist = self.decoder(flatten_posterior, flatten_rnn_hiddens)
        reward_dist = self.reward_model(flatten_posterior, flatten_rnn_hiddens)
        pcont_dist = self.discount_model(flatten_posterior, flatten_rnn_hiddens)

        obs_loss = self._obs_loss(obs_dist, obs[:-1])
        reward_loss = self._obs_loss(reward_dist, rewards[1:])
        pcont_loss = self._obs_loss(pcont_dist, nonterms[1:])
        kl_loss, prior_dist, post_dist = self._kl_loss(prior_logits, posterior_logits)

        model_loss = (
            self.kl_loss_scale * kl_loss
            + reward_loss
            + obs_loss
            + self.discount_loss_scale * pcont_loss
        )

        return (
            model_loss,
            kl_loss,
            obs_loss,
            reward_loss,
            pcont_loss,
            prior_dist,
            post_dist,
            flatten_posterior,
            flatten_rnn_hiddens,
        )

    def _actor_loss(
        self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy
    ):
        # λ-returnを計算
        def compute_return(
            reward: torch.Tensor,
            value: torch.Tensor,
            discount: torch.Tensor,
            bootstrap: torch.Tensor,
            lambda_: float,
        ):
            """
            Compute the discounted reward for a batch of data.
            reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
            Bootstrap is [batch, 1]
            """
            next_values = torch.cat([value[1:], bootstrap[None]], 0)
            target = reward + discount * next_values * (1 - lambda_)
            timesteps = list(range(reward.shape[0] - 1, -1, -1))
            outputs = []
            accumulated_reward = bootstrap
            for t in timesteps:
                inp = target[t]
                discount_factor = discount[t]
                accumulated_reward = (
                    inp + discount_factor * lambda_ * accumulated_reward
                )
                outputs.append(accumulated_reward)
            returns = torch.flip(torch.stack(outputs), [0])
            return returns

        lambda_returns = compute_return(
            imag_reward[:-1],
            imag_value[:-1],
            discount_arr[:-1],
            bootstrap=imag_value[-1],
            lambda_=self.lambda_,
        )
        advantage = (lambda_returns - imag_value[:-1]).detach()
        objective = imag_log_prob[1:].unsqueeze(-1) * advantage

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
        actor_loss = -torch.sum(
            torch.mean(
                discount * (objective + self.actor_entropy_scale * policy_entropy),
                dim=1,
            )
        )
        return actor_loss, discount, lambda_returns

    def _value_loss(self, states, rnn_hiddens, discount, lambda_returns):
        with torch.no_grad():
            value_states = states[:-1].detach()
            value_rnn_hiddens = rnn_hiddens[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.value_model(value_states, value_rnn_hiddens)
        value_loss = -torch.mean(
            value_discount * value_dist.log_prob(value_target).unsqueeze(-1)
        )
        return value_loss

    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss

    def _kl_loss(self, prior_logit, posterior_logit):
        prior_dist = self.rssm.get_dist(prior_logit)
        posterior_dist = self.rssm.get_dist(posterior_logit)
        alpha = self.kl_balance_scale
        kl_lhs = torch.mean(
            torch.distributions.kl.kl_divergence(
                self.rssm.get_dist(posterior_logit.detach()), prior_dist
            )
        )

        kl_rhs = torch.mean(
            torch.distributions.kl.kl_divergence(
                posterior_dist, self.rssm.get_dist(posterior_logit.detach())
            )
        )
        # KL誤差がfree_nats以下の時は無視
        kl_loss = (
            alpha * kl_lhs.clamp(min=self.free_nats).mean()
            + (1 - alpha) * kl_rhs.clamp(min=self.free_nats).mean()
        )
        return kl_loss, prior_dist, posterior_dist

    def reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss

    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

    def update_target(self):
        for param, target in zip(
            self.value_model.parameters(), self.target_value_model.parameters()
        ):
            target.data.copy(param.data)

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
        self.target_value_model = ValueModel(
            state_dim=state_dim, rnn_hidden_dim=rnn_hidden_dim
        )
        self.target_value_model.load_state_dict(self.value_model.state_dict())
        self.reward_model = RewardModel(
            state_dim=state_dim, rnn_hidden_dim=rnn_hidden_dim
        )
        self.discount_model = DiscountModel(
            state_dim=state_dim, rnn_hidden_dim=rnn_hidden_dim
        )

        self.encoder = ObsEncoder(self.obs_shape, embedding_size)
        self.decoder = ObsDecoder(self.obs_shape, embedding_size)

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
        self.model_optimizer = optim.Adam(
            self.model_params, lr=config.model_lr, eps=config.eps
        )
        self.value_optimizer = optim.Adam(
            self.value_model.parameters(), lr=config.critic_lr, eps=config.eps
        )
        self.action_optimizer = optim.Adam(
            self.action_model.parameters(), lr=config.actor_lr, eps=config.eps
        )

    def _agent_initialize(self):
        self.agent = Agent(self.encoder, self.rssm, self.action_model)

    def _print_summary(self):
        pass

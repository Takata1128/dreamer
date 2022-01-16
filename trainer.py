import os
import torch
import numpy as np
from torch import optim
from torch import nn
from utils import FreezeParameters
from replay_buffer import TransitionBuffer
from models.rssm import RecurrentStateSpaceModel
from models.observation_model import ObsEncoder, ObsDecoder
from models.reward_model import RewardModel
from models.action_model import ActionModel
from models.value_model import ValueModel
from models.discount_model import DiscountModel


class Trainer(object):
    """
    世界モデルとエージェントの学習を管理
    """

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
        self.discount = config.gamma
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

    def train_batch(self, train_metrics):
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
            # バッファから経験をサンプル ([chunk_length,batch_size,*])
            # obs: [t,t+chunk_length]
            # action,rewards,nonterms: [t-1,t+chunk_length-1]

            observations, actions, rewards, terminals = self.replay_buffer.sample(
                batch_size=self.batch_size, chunk_length=self.chunk_length
            )

            # データの前処理
            observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            rewards = torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)
            non_terminals = torch.tensor(
                1 - terminals, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)

            ### 世界モデルの更新 ###
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
                + list(self.discount_model.parameters())
            )
            nn.utils.clip_grad_norm_(model_params, self.clip_grad_norm)
            self.model_optimizer.step()

            ### Actor-Criticモデルの更新 ###
            # 世界モデルの学習の際に推論したrssm状態を用いる
            actor_loss, value_loss, target_info = self.actor_critic_loss(
                states, rnn_hiddens
            )

            self.actor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            actor_loss.backward()
            value_loss.backward()

            nn.utils.clip_grad_norm_(
                self.action_model.parameters(), self.clip_grad_norm
            )
            nn.utils.clip_grad_norm_(self.value_model.parameters(), self.clip_grad_norm)

            self.actor_optimizer.step()
            self.value_optimizer.step()

            with torch.no_grad():
                prior_ent = torch.mean(prior_dist.entropy())
                posterior_ent = torch.mean(post_dist.entropy())

            # メトリクスの保存
            prior_ent_l.append(prior_ent.item())
            post_ent_l.append(posterior_ent.item())
            actor_l.append(actor_loss.item())
            value_l.append(value_loss.item())
            obs_l.append(obs_loss.item())
            model_l.append(model_loss.item())
            reward_l.append(reward_loss.item())
            kl_l.append(kl_loss.item())
            pcont_l.append(pcont_loss.item())
            mean_targ.append(target_info["mean_targ"])
            min_targ.append(target_info["min_targ"])
            max_targ.append(target_info["max_targ"])
            std_targ.append(target_info["std_targ"])

        train_metrics["model_loss"] = np.mean(model_l)
        train_metrics["kl_loss"] = np.mean(kl_l)
        train_metrics["reward_loss"] = np.mean(reward_l)
        train_metrics["obs_loss"] = np.mean(obs_l)
        train_metrics["value_loss"] = np.mean(value_l)
        train_metrics["actor_loss"] = np.mean(actor_l)
        train_metrics["prior_entropy"] = np.mean(prior_ent_l)
        train_metrics["posterior_entropy"] = np.mean(post_ent_l)
        train_metrics["pcont_loss"] = np.mean(pcont_l)
        train_metrics["mean_targ"] = np.mean(mean_targ)
        train_metrics["min_targ"] = np.mean(min_targ)
        train_metrics["max_targ"] = np.mean(max_targ)
        train_metrics["std_targ"] = np.mean(std_targ)

        return train_metrics

    def actor_critic_loss(self, states, rnn_hiddens):
        """
        Actor Criticのロス計算
        """

        # 今までの勾配を遮断
        with torch.no_grad():
            # [chunk,batch,*] -> [chunk-1*batch,*]
            flatten_states = states[:-1].view(-1, self.state_dim).detach()
            flatten_rnn_hiddens = (
                rnn_hiddens[:-1].view(-1, self.rnn_hidden_dim).detach()
            )

        # 世界モデルのパラメータは凍結
        with FreezeParameters(
            [
                self.encoder,
                self.decoder,
                self.rssm,
                self.reward_model,
                self.discount_model,
            ]
        ):
            # 観測データから並列に想像上の軌道を生成
            (
                imaginated_states,
                imaginated_rnn_hiddens,
                imag_action_log_prob,
                policy_entropy,
            ) = self.rollout_imagination(
                self.horizon, flatten_states, flatten_rnn_hiddens
            )

        # [horizon,chunk-1*batch,*]

        # Actor以外のパラメータは凍結
        with FreezeParameters(
            [
                self.encoder,
                self.decoder,
                self.rssm,
                self.reward_model,
                self.discount_model,
                self.value_model,
                self.target_value_model,
            ]
        ):
            # 想像上の軌道におけるreward,value,discount
            imag_reward_dist = self.reward_model(
                imaginated_states, imaginated_rnn_hiddens
            )
            imag_reward = imag_reward_dist.mean
            imag_value_dist = self.target_value_model(
                imaginated_states, imaginated_rnn_hiddens
            )
            imag_value = imag_value_dist.mean
            discount_dist = self.discount_model(
                imaginated_states, imaginated_rnn_hiddens
            )
            discount_arr = self.discount * torch.round(discount_dist.base_dist.probs)

        # λ-returnを計算
        def compute_return(
            reward: torch.Tensor,
            value: torch.Tensor,
            discount: torch.Tensor,
            bootstrap: torch.Tensor,
            lambda_: float,
        ):
            """
            バッチデータごとの割引報酬を計算
            reward,value,discount,return:[horizon-1,batch,1]
            bootstrap:[horizon-1,1]
            """
            next_values = torch.cat([value[1:], bootstrap[None]], 0)
            target = reward + discount * next_values * (1 - lambda_)
            timesteps = list(range(reward.shape[0] - 1, -1, -1))
            outputs = []
            accumulated_reward = bootstrap
            # 逆から計算
            for t in timesteps:
                inp = target[t]
                discount_factor = discount[t]
                accumulated_reward = (
                    inp + discount_factor * lambda_ * accumulated_reward
                )
                outputs.append(accumulated_reward)
            # 最後で反転
            returns = torch.flip(torch.stack(outputs), [0])
            return returns

        lambda_returns = compute_return(
            imag_reward[:-1],
            imag_value[:-1],
            discount_arr[:-1],
            bootstrap=imag_value[-1],
            lambda_=self.lambda_,
        )

        # 割引率配列の設定
        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)

        # ロスを計算
        actor_loss = self._actor_loss(
            lambda_returns, imag_value, discount, imag_action_log_prob, policy_entropy
        )
        value_loss = self._value_loss(
            imaginated_states, imaginated_rnn_hiddens, discount, lambda_returns
        )

        # lambda-targetのもろもろ
        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item()
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()

        target_info = {
            "min_targ": min_targ,
            "max_targ": max_targ,
            "std_targ": std_targ,
            "mean_targ": mean_targ,
        }

        return actor_loss, value_loss, target_info

    def rollout_imagination(
        self, imagination_horizon, flatten_states, flatten_rnn_hiddens
    ):
        """
        RSSMにより想像上の軌道を生成
        """
        imaginated_states = []
        imaginated_rnn_hiddens = []
        action_entropy = []
        imag_action_log_probs = []

        for t in range(imagination_horizon):
            # priorで未来予測しつつ行動
            action, action_dist = self.action_model(
                flatten_states.detach(), flatten_rnn_hiddens.detach()
            )
            flatten_states_prior_logits, flatten_rnn_hiddens = self.rssm.prior(
                flatten_states, action, flatten_rnn_hiddens
            )
            flatten_states = self.rssm.get_stoch_state(flatten_states_prior_logits)

            # 保存
            imaginated_states.append(flatten_states)
            imaginated_rnn_hiddens.append(flatten_rnn_hiddens)
            action_entropy.append(action_dist.entropy())
            imag_action_log_probs.append(
                action_dist.log_prob(torch.round(action.detach()))
            )

        imaginated_states = torch.stack(imaginated_states, dim=0)
        imaginated_rnn_hiddens = torch.stack(imaginated_rnn_hiddens, dim=0)
        imag_action_log_probs = torch.stack(imag_action_log_probs, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)

        return (
            imaginated_states,
            imaginated_rnn_hiddens,
            imag_action_log_probs,
            action_entropy,
        )

    def _actor_loss(
        self, lambda_returns, imag_value, discount, imag_action_log_prob, policy_entropy
    ):
        """
        Actor-Loss: E [ sum_t=1..H-1{- ln pψ(a't | z't) sg(Vt^λ - vξ(z't)) - H[at|z't]}]
        (Reinforceのみ)
        """

        advantage = (lambda_returns - imag_value[:-1]).detach()
        objective = imag_action_log_prob[1:].unsqueeze(-1) * advantage

        policy_entropy = policy_entropy[1:].unsqueeze(-1)
        # ロスにdiscountをかけるのが何故かよくわかってないが、公式実装もそうしてるっぽいのでそうする
        # より将来の確定的でないものを安くしてる？
        actor_loss = -torch.sum(
            torch.mean(
                discount * (objective + self.actor_entropy_scale * policy_entropy),
                dim=1,
            )
        )
        return actor_loss

    def _value_loss(self, states, rnn_hiddens, discount, lambda_returns):
        """
        Critic-Loss
        valueの対数尤度ロス 論文上はMSEだけど等価（だと思ってる）
        """

        with torch.no_grad():
            value_states = states[:-1].detach()
            value_rnn_hiddens = rnn_hiddens[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.value_model(value_states, value_rnn_hiddens)
        # ここもdiscountをかけてる
        value_loss = -torch.mean(
            value_discount * value_dist.log_prob(value_target).unsqueeze(-1)
        )
        return value_loss

    def representation_loss(self, obs, actions, rewards, nonterms):
        """
        世界モデルのロス計算
        """
        # 観測のエンコード
        embed = self.encoder(obs)

        # サンプルしたデータからrssm状態を構築
        (
            prior_logits,
            posterior_logits,
            states,
            rnn_hiddens,
        ) = self.predict_states(embed, actions, nonterms)

        # 観測、報酬、episode終了の0-1を再構成
        obs_dist = self.decoder(states[:-1], rnn_hiddens[:-1])
        reward_dist = self.reward_model(states[:-1], rnn_hiddens[:-1])
        pcont_dist = self.discount_model(states[:-1], rnn_hiddens[:-1])

        # ロスを計算
        obs_loss = self._obs_loss(obs_dist, obs[:-1])  # 復元した状態と観測の間のロス
        reward_loss = self._reward_loss(reward_dist, rewards[1:])  # 次ステップの報酬との比較
        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])  # 次ステップのdoneとの比較
        kl_loss, prior_dist, post_dist = self._kl_loss(prior_logits, posterior_logits)

        model_loss = (
            self.kl_loss_scale * kl_loss
            + self.reward_loss_scale * reward_loss
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
            states,
            rnn_hiddens,
        )

    def predict_states(self, embedded_obs, action, nonterms):
        """
        観測データを取り込んでchunk_length分の状態表現を推論
        """
        # 低次元の状態表現保持のためのTensor
        states = []
        rnn_hiddens = []
        prior_logits = []
        posterior_logits = []

        # rssm状態の初期値
        state = torch.zeros(self.batch_size, self.state_dim, device=self.device)
        rnn_hidden = torch.zeros(
            self.batch_size, self.rnn_hidden_dim, device=self.device
        )

        prev_state = state
        for l in range(self.chunk_length):
            # エピソード終了を考慮しつつ行動と観測を受け取りながら推論
            prev_action = action[l] * nonterms[l]
            prior_logit, posterior_logit, rnn_hidden = self.rssm(
                prev_state, prev_action, rnn_hidden, embedded_obs[l], nonterms[l]
            )
            prev_state = self.rssm.get_stoch_state(posterior_logit)

            # 保存
            states.append(prev_state)
            rnn_hiddens.append(rnn_hidden)
            prior_logits.append(prior_logit)
            posterior_logits.append(posterior_logit)

        states = torch.stack(states, dim=0)
        rnn_hiddens = torch.stack(rnn_hiddens, dim=0)
        prior_logits = torch.stack(prior_logits, dim=0)
        posterior_logits = torch.stack(posterior_logits, dim=0)

        return prior_logits, posterior_logits, states, rnn_hiddens

    def _obs_loss(self, obs_dist, obs):
        """
        観測の対数尤度ロス
        """
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss

    def _kl_loss(self, prior_logit, posterior_logit):
        """
        prior-posterior間のKL-divergenceロス
        kl-balancingあり
        """
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
                posterior_dist, self.rssm.get_dist(prior_logit.detach())
            )
        )
        # KL誤差がfree_nats以下の時は無視
        # kl_lhs = torch.max(kl_lhs, kl_lhs.new_full(kl_lhs.size(), self.free_nats))
        # kl_rhs = torch.max(kl_rhs, kl_rhs.new_full(kl_rhs.size(), self.free_nats))
        kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs
        return kl_loss, prior_dist, posterior_dist

    def _reward_loss(self, reward_dist, rewards):
        """
        報酬の対数尤度ロス
        """
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss

    def _pcont_loss(self, pcont_dist, nonterms):
        """
        discount(0-1)の対数尤度ロス
        """
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

    def update_target(self):
        """
        valueモデルをtargetネットワークと同期
        """
        for param, target in zip(
            self.value_model.parameters(), self.target_value_model.parameters()
        ):
            target.data.copy_(param.data)

    def save_model(self, iter):
        """
        全モデルの保存
        """
        save_dict = self.get_save_dict()
        model_dir = self.config.model_dir
        save_path = os.path.join(model_dir, "models_%d.pth" % iter)
        torch.save(save_dict, save_path)

    def get_save_dict(self):
        """
        全モデルのパラメータ
        """
        return {
            "RSSM": self.rssm.state_dict(),
            "Encoder": self.encoder.state_dict(),
            "Decoder": self.decoder.state_dict(),
            "RewardModel": self.reward_model.state_dict(),
            "ActionModel": self.action_model.state_dict(),
            "ValueModel": self.value_model.state_dict(),
            "DiscountModel": self.discount_model.state_dict(),
        }

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
            action_size,
            state_dim,
            rnn_hidden_dim,
            category_size,
            class_size,
        ).to(self.device)
        self.action_model = ActionModel(
            state_dim=state_dim, rnn_hidden_dim=rnn_hidden_dim, action_dim=action_size
        ).to(self.device)
        self.value_model = ValueModel(
            state_dim=state_dim, rnn_hidden_dim=rnn_hidden_dim
        ).to(self.device)
        self.target_value_model = ValueModel(
            state_dim=state_dim, rnn_hidden_dim=rnn_hidden_dim
        ).to(self.device)
        self.target_value_model.load_state_dict(self.value_model.state_dict())
        self.reward_model = RewardModel(
            state_dim=state_dim, rnn_hidden_dim=rnn_hidden_dim
        ).to(self.device)
        self.discount_model = DiscountModel(
            state_dim=state_dim, rnn_hidden_dim=rnn_hidden_dim
        ).to(self.device)

        self.encoder = ObsEncoder(self.obs_shape, embedding_size).to(self.device)
        self.decoder = ObsDecoder(self.obs_shape, state_dim + rnn_hidden_dim).to(
            self.device
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
            + list(self.discount_model.parameters())
        )
        self.model_optimizer = optim.Adam(self.model_params, lr=config.model_lr)
        self.value_optimizer = optim.Adam(
            self.value_model.parameters(), lr=config.critic_lr
        )
        self.actor_optimizer = optim.Adam(
            self.action_model.parameters(), lr=config.actor_lr
        )

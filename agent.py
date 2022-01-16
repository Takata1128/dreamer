import torch
import numpy as np


class Agent:
    """
    ActionModelに基づき行動を決定する
    """

    def __init__(self, encoder, rssm, action_model, config):
        self.encoder = encoder
        self.rssm = rssm
        self.action_model = action_model
        self.device = next(self.action_model.parameters()).device
        self.action_size = config.action_size
        self.prev_state = torch.zeros(1, rssm.state_dim, device=self.device)
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)
        self.prev_action = torch.zeros(1, self.action_size, device=self.device)

        self.train_noise = config.train_noise
        self.expl_min = config.expl_min
        self.expl_decay = config.expl_decay

    def __call__(self, obs, nonterm, step, training=True):
        """
        actionとその分布を返却（勾配なし）
        """
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(
                0
            )
            # 観測を低次元に変換し,posteriorからのサンプルをActionModelに入力して行動を決定
            embedded_obs = self.encoder(obs)
            _, posterior_logit, rnn_hidden = self.rssm(
                self.prev_state,
                self.prev_action,
                self.rnn_hidden,
                embedded_obs,
                nonterm,
            )
            # posterior_logit = self.rssm.posterior(self.rnn_hidden, embedded_obs)
            state = self.rssm.get_stoch_state(posterior_logit)
            action, action_dist = self.action_model(state, rnn_hidden)
            # train時はたまにランダム行動をとる
            if training:
                action = self._add_exploration(action, step).detach()
            # # 次のステップのためにRNNの隠れ状態を更新
            # _, self.rnn_hidden = self.rssm.prior(
            #     state, action, self.rnn_hidden, nonterm
            # )
            self.prev_state = state
            self.prev_action = action
            self.rnn_hidden = rnn_hidden

        return action, action_dist

    def _add_exploration(self, action, itr):
        """
        たまにランダム行動（Epsilon-greedy）
        参考実装どおりに実装
        行動自体もともとlogitに従ったサンプルなのである程度探索できそうな気がするけど...
        """
        expl_amount = self.train_noise
        expl_amount = expl_amount - itr / self.expl_decay
        expl_amount = max(self.expl_min, expl_amount)

        if np.random.uniform(0, 1) < expl_amount:
            index = torch.randint(
                0, self.action_size, action.shape[:-1], device=action.device
            )
            action = torch.zeros_like(action)
            action[:, index] = 1
        return action

    def reset(self):
        self.prev_state = torch.zeros(1, self.rssm.state_dim, device=self.device)
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)
        self.prev_action = torch.zeros(1, self.action_size, device=self.device)

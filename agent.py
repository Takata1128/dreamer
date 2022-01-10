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
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)
        self.action_size = config.action_size

        self.train_noise = config.train_noise
        self.expl_min = config.expl_min
        self.expl_decay = config.expl_decay

    def __call__(self, obs, nonterm, training=True):
        """
        actionとその分布を返却（勾配なし）
        """
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            # 観測を低次元に変換し,posteriorからのサンプルをActionModelに入力して行動を決定
            embedded_obs = self.encoder(obs)
            posterior_logit, rnn_hidden = self.rssm.posterior(
                self.rnn_hidden, embedded_obs
            )
            # state = state_posterior.sample()
            state = self.rssm.get_stoch_state(posterior_logit)
            action, action_dist = self.action_model(
                state, self.rnn_hidden, training=training
            )

            # 次のステップのためにRNNの隠れ状態を更新
            _, self.rnn_hidden = self.rssm.prior(
                state, action, self.rnn_hidden, nonterm
            )

        return action, action_dist

    def add_exploration(self, action, itr):
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
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)

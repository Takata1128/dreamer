import torch
from utils import preprocess_obs


class Agent:
    """
    ActionModelに基づき行動を決定
    """

    def __init__(self, encoder, rssm, action_model):
        self.encoder = encoder
        self.rssm = rssm
        self.action_model = action_model

        self.device = next(self.action_model.parameters()).device
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs, training=True):
        # preprocessとchannel-first化
        # obs = preprocess_obs(obs)
        obs = torch.as_tensor(obs, device=self.device)
        # obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)
        obs = obs.unsqueeze(0)

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
            _, self.rnn_hidden = self.rssm.prior(state, action, self.rnn_hidden)

        return action.squeeze().cpu().numpy()

    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)

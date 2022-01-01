import torch
from torch import nn
from torch import functional as F
from torch.distributions import Normal


class RecurrentStateSpaceModel(nn.Module):
    """
    1ステップ先の未来の状態表現を予測する
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        rnn_hidden_dim,
        hidden_dim=200,
        min_stddev=0.1,
        act=F.elu,
    ):
        super(RecurrentStateSpaceModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc_state_action = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_rnn_hidden_embedded_obs = nn.Linear(rnn_hidden_dim + 1024, hidden_dim)
        self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_posterior = nn.Linear(hidden_dim, state_dim)
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self._min_stddev = min_stddev
        self.act = act

    def forward(self, state, action, rnn_hidden, embedded_next_obs):
        """
        h_t+1 = f(h_t,s_t,a_t)
        prior p(s_t+1 | h_t+1) と posterior p(s_t+1 | h_t+1, o_t+1)を返す
        """
        next_state_prior, rnn_hidden = self.prior(state, action, rnn_hidden)
        next_state_posterior = self.posterior(rnn_hidden, embedded_next_obs)
        return next_state_prior, next_state_posterior

    def prior(self, state, action, rnn_hidden):
        """
        prior p(s_t+1|h_t+1)
        """
        hidden = self.act(self.fc_state_action(torch.cat([state, action], dim=1)))
        rnn_hidden = self.rnn(hidden, rnn_hidden)
        hidden = self.act(self.fc_rnn_hidden(rnn_hidden))

        mean = self.fc_state_mean_prior(hidden)
        stddev = F.softplus(self.fc_state_stddev_prior(hidden)) + self._min_stddev
        return Normal(mean, stddev), rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs):
        """
        posterior q(s_t|h_t,o_t)
        """
        hidden = self.act(
            self.fc_rnn_hidden_embedded_obs(
                torch.cat([rnn_hidden, embedded_obs], dim=1)
            )
        )
        mean = self.fc_state_mean_posterior(hidden)
        stddev = F.softplus(self.fc_state_stddev_posterior(hidden)) + self._min_stddev
        return Normal(mean, stddev)

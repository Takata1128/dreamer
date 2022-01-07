import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class ActionModel(nn.Module):
    """
    低次元の状態表現から行動を出力
    """

    def __init__(
        self,
        state_dim,
        rnn_hidden_dim,
        action_dim,
        hidden_dim=400,
        act=F.elu,
        min_stddev=1e-4,
        init_stddev=5.0,
    ):
        super(ActionModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_action = nn.Linear(hidden_dim, action_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_stddev = nn.Linear(hidden_dim, action_dim)
        self.act = act
        self.min_stddev = min_stddev
        self.init_stddev = init_stddev

    def forward(self, state, rnn_hidden, training=False):
        """
        training=True: NNのパラメータに関して微分可能なかたちの行動のサンプル
        training=False: 行動の確率分布の平均値
        """
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        hidden = self.act(self.fc4(hidden))

        logits = self.fc_action(hidden)
        action_dist = torch.distributions.OneHotCategorical(logits=logits)
        action = action_dist.sample()
        action = action + action_dist.probs - action_dist.probs.detach()
        return action, action_dist

        # Dreamerの実装に合わせて平均と分散に対する変換
        mean = self.fc_mean(hidden)
        mean = 5.0 * torch.tanh(mean / 5.0)

        stddev = self.fc_stddev(hidden)
        stddev = F.softplus(stddev + self.init_stddev) + self.min_stddev

        if training:
            action = torch.tanh(Normal(mean, stddev).rsample())
        else:
            action = torch.tanh(mean)
        return action

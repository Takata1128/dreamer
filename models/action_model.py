import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class ActionModel(nn.Module):
    """
    低次元の状態表現から行動を出力
    """

    def __init__(
        self, state_dim, rnn_hidden_dim, action_dim, hidden_dim=400, act=F.elu
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

    def forward(self, state, rnn_hidden, training=False):
        """
        微分可能な行動のサンプルとその分布を出力
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

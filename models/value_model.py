import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td


class ValueModel(nn.Module):
    """
    低次元の状態表現から状態価値を出力
    """

    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=400, act=F.elu):
        super(ValueModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state, rnn_hidden):
        """
        Valueを予測する正規分布を返す
        """
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=2)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        state_value = self.fc4(hidden)
        return td.independent.Independent(td.Normal(state_value, 1), 1)

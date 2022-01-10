import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td


class RewardModel(nn.Module):
    """
    p(r_t|s_t,h_t)
    低次元の状態表現から報酬を予測
    """

    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=400, act=F.elu):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state, rnn_hidden):
        """
        報酬を予測する正規分布を返す
        """
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=2)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        reward = self.fc4(hidden)
        return td.independent.Independent(td.Normal(reward, 1), 1)

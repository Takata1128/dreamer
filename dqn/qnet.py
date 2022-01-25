import torch
import torch.nn as nn
import numpy as np
import random


class QNetwork(nn.Module):
    """
    Dueling Networkを用いたQ関数
    """

    def __init__(
        self,
        state_shape,
        n_action,
        node_size=200,
        depth=16,
        kernel=3,
        activation=nn.ELU,
    ):
        super(QNetwork, self).__init__()

        self.state_shape = state_shape
        self.d = depth
        self.k = kernel

        self.n_action = n_action

        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_shape[0], depth, kernel),
            activation(),
            nn.Conv2d(depth, 2 * depth, kernel),
            activation(),
            nn.Conv2d(2 * depth, 4 * depth, kernel),
            activation(),
        )

        self.fc_state = nn.Sequential(
            nn.Linear(self.embed_size, node_size), nn.ELU(), nn.Linear(node_size, 1)
        )
        self.fc_advantage = nn.Sequential(
            nn.Linear(self.embed_size, node_size),
            nn.ELU(),
            nn.Linear(node_size, n_action),
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        feature = self.conv_layers(obs)
        feature = torch.reshape(feature, (batch_size, -1))
        state_values = self.fc_state(feature)
        advantage = self.fc_advantage(feature)

        # 行動価値=状態価値+アドバンテージ アドバンテージの平均を引いて安定化
        action_values = (
            state_values + advantage - torch.mean(advantage, dim=1, keepdim=True)
        )

        return action_values

    def act(self, obs, epsilon):
        """
        epsilon-greedyに行動選択
        """
        if random.random() < epsilon:
            action = random.randrange(self.n_action)
        else:
            with torch.no_grad():
                action = torch.argmax(self.forward(obs.unsqueeze(0))).item()
        return action

    @property
    def embed_size(self):
        """
        畳み込みサイズ計算
        """
        conv1_shape = conv_out_shape(self.state_shape[1:], 0, self.k, 1)
        conv2_shape = conv_out_shape(conv1_shape, 0, self.k, 1)
        conv3_shape = conv_out_shape(conv2_shape, 0, self.k, 1)
        embed_size = int(4 * self.d * np.prod(conv3_shape).item())
        return embed_size


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

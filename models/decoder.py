import torch
from torch import nn
from torch.nn import functional as F


class ObservationModel(nn.Module):
    def __init__(self, output_shape, embed_size, info):
        """
        :param output_shape: tuple containing shape of output obs
        :param embed_size: the size of input vector, for dreamerv2 : modelstate
        """
        super(ObservationModel, self).__init__()
        c, h, w = output_shape
        activation = nn.ELU
        d = info["depth"]
        k = info["kernel"]
        conv1_shape = conv_out_shape(output_shape[1:], 0, 3, 1)
        conv2_shape = conv_out_shape(conv1_shape, 0, 3, 1)
        conv3_shape = conv_out_shape(conv2_shape, 0, 3, 1)
        self.conv_shape = (4 * d, *conv3_shape)
        self.output_shape = output_shape
        if embed_size == np.prod(self.conv_shape).item():
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(embed_size, np.prod(self.conv_shape).item())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4 * d, 2 * d, k, 1),
            activation(),
            nn.ConvTranspose2d(2 * d, d, k, 1),
            activation(),
            nn.ConvTranspose2d(d, c, k, 1),
        )

    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.decoder(x)
        mean = torch.reshape(x, (*batch_shape, *self.output_shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.output_shape))
        return obs_dist


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(
        output_padding(h_in[i], conv_out[i], padding, kernel_size, stride)
        for i in range(len(h_in))
    )


class ObservationModel(nn.Module):
    """
    p(o_t|s_t,h_t)
    低次元の状態表現から画像を再構成
    """

    def __init__(self, state_dim, rnn_hidden_dim):
        super(ObservationModel, self).__init__()
        self.fc = nn.Linear(state_dim + rnn_hidden_dim, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def forward(self, state, rnn_hidden):
        hidden = self.fc(torch.cat([state, rnn_hidden], dim=1))
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.relu(self.dc1(hidden))
        hidden = F.relu(self.dc2(hidden))
        hidden = F.relu(self.dc3(hidden))
        obs = self.dc4(hidden)
        return obs

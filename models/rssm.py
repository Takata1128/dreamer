import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class RecurrentStateSpaceModel(nn.Module):
    """
    1ステップ先の未来の状態表現を予測する
    """

    def __init__(
        self,
        node_size,
        embedding_size,
        action_dim,
        state_dim,
        rnn_hidden_dim,
        category_size=20,
        class_size=20,
        act=F.elu,
    ):
        super(RecurrentStateSpaceModel, self).__init__()
        self.node_size = node_size
        self.embedding_size = embedding_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.category_size = category_size
        self.class_size = class_size

        # prior
        self.fc_state_action = nn.Linear(state_dim + action_dim, node_size)
        self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, node_size)
        self.fc_prior = nn.Linear(node_size, state_dim)

        # posterior
        self.fc_rnn_hidden_embedding = nn.Linear(
            rnn_hidden_dim + embedding_size, node_size
        )
        self.fc_posterior = nn.Linear(node_size, state_dim)

        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.act = act

    def forward(self, state, action, rnn_hidden, embedded_next_obs, nonterms=True):
        """
        h_t+1 = f(h_t,s_t,a_t)
        prior p(s_t+1 | h_t+1) と posterior p(s_t+1 | h_t+1, o_t+1)のロジットを返す
        """
        ns_prior_logit, rnn_hidden = self.prior(state, action, rnn_hidden, nonterms)
        ns_posterior_logit = self.posterior(rnn_hidden, embedded_next_obs)
        return (
            ns_prior_logit,
            ns_posterior_logit,
            rnn_hidden,
        )

    def prior(self, state, action, rnn_hidden, nonterms=True):
        """
        h_t+1 = f(h_t,s_t,a_t)
        prior p(s_t+1|h_t+1)
        """
        hidden = self.act(
            self.fc_state_action(torch.cat([state * nonterms, action], dim=-1))
        )
        rnn_hidden = self.rnn(hidden, rnn_hidden * nonterms)
        hidden = self.act(self.fc_rnn_hidden(rnn_hidden))
        prior_logit = self.fc_prior(hidden)
        return prior_logit, rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs):
        """
        posterior q(s_t|h_t,o_t)
        """
        hidden = self.act(
            self.fc_rnn_hidden_embedding(torch.cat([rnn_hidden, embedded_obs], dim=-1))
        )
        posterior_logit = self.fc_posterior(hidden)
        return posterior_logit

    def get_dist(self, logit):
        """
        logitをrssm状態(z)の分布にして返却（KL-divの計算に使用）
        """
        shape = logit.shape
        logit = torch.reshape(
            logit, shape=(*shape[:-1], self.category_size, self.class_size)
        )
        # [batch:[batch,category], event:[class]] → [batch:[batch], event:[category,class]]
        return torch.distributions.Independent(
            torch.distributions.OneHotCategoricalStraightThrough(logits=logit), 1
        )

    def get_stoch_state(self, logit):
        """
        logitから（平坦化した）rssm状態(z)に
        """
        shape = logit.shape
        logit = torch.reshape(
            logit, shape=(*shape[:-1], self.category_size, self.class_size)
        )
        dist = torch.distributions.OneHotCategorical(logits=logit)
        # straight-through
        stoch = dist.sample()
        stoch += dist.probs - dist.probs.detach()
        # [batch,category,class] -> [batch,category*class]
        return torch.flatten(stoch, start_dim=-2, end_dim=-1)

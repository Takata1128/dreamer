class Trainer(object):
    def __init__(self, config, device):
        pass

    def collect_seed_episodes(self, env):
        s, done = env.reset(), False
        for i in range(self.seed_steps):
            a = env.action_space.sample()
            ns, r, done, _ = env.step(a)
            self.buffer.add(s, a, r, done)
            if done:
                s, done = env.reset(), False
            else:
                s = ns

    def train_batch(self, train_metrics):
        pass

    def actor_critic_loss(self, posterior):
        pass

    def representation_loss(self, obs, actions, rewards, nonterms):
        pass

    def _actor_loss(
        self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy
    ):
        pass

    def _value_loss(self, imag_modelstates, discount, lambda_returns):
        pass

    def _obs_loss(self, obs_dist, obs):
        pass

    def _kl_loss(self, prior, posterior):
        pass

    def reward_loss(self, reward_dist, rewards):
        pass

    def _pcont_loss(self, pcont_dist, nonterms):
        pass

    def update_target(self):
        pass

    def save_model(self, iter):
        pass

    def get_save_dict(self):
        pass

    def load_save_dict(self, saved_dict):
        pass

    def _model_initialize(self, config):
        pass

    def _optim_initialize(self, config):
        pass

    def _print_summary(self):
        pass

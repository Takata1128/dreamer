import minatar
import gym
import numpy as np


class GymMinAtar(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, env_name, display_time=50):
        self.display_time = display_time
        self.env_name = env_name
        self.env = minatar.Environment(env_name)
        self.minimal_actions = self.env.minimal_action_set()
        h, w, c = self.env.state_shape()
        self.action_space = gym.spaces.Discrete(len(self.minimal_actions))
        self.observation_space = gym.spaces.MultiBinary((c, h, w))

    def reset(self):
        self.env.reset()
        return self.env.state().transpose(2, 0, 1)

    def step(self, index):
        """index is the action id, considering only the set of minimal actions"""
        action = self.minimal_actions[index]
        r, terminal = self.env.act(action)
        self.game_over = terminal
        return self.env.state().transpose(2, 0, 1), r, terminal, {}

    def seed(self, seed="None"):
        self.env = minatar.Environment(self.env_name, random_seed=seed)

    def render(self, mode="human"):
        if mode == "rgb_array":
            return self.env.state()
        elif mode == "human":
            self.env.display_state(self.display_time)

    def close(self):
        if self.env.visualized:
            self.env.close_display()
        return 0


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference

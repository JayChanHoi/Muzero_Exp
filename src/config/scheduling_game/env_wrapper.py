from collections import deque

import numpy as np

from ...core.game import Game, Action

class SchedulingGameWrapper(Game):
    def __init__(self, env, k: int, discount: float):
        """

        :param env: instance of gym environment
        :param k: no. of observations to stack
        """
        super().__init__(env, env.action_space.n, discount, k=k)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.env = env

    def legal_actions(self):
        return [Action(_) for _ in self.env.get_available_actions().nonzero().squeeze().tolist()]

    def get_obs_info(self):
        return self.env.get_obs_info()

    def step(self, action):

        obs, reward, done, _ = self.env.step(action)

        self.rewards.append(reward)
        self.history.append(action)
        self.obs_history.append(obs)

        return self.obs(len(self.rewards), self.k), reward, done, _

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.rewards = []
        self.history = []
        self.obs_history = []

        for _ in range(self.k):
            self.obs_history.append(obs)

        return self.obs(0, self.k)

    def close(self):
        self.env.close()

    def obs(self, i, k):
        return self.obs_history[i:i + k]

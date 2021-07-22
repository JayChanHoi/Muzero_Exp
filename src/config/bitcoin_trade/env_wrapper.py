from collections import deque

import numpy as np

# from ...core.game import Game, Action
from ...core.game import Game

class BitcoinTradeGameWrapper(Game):
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
        legal_action_list = self.env.get_action_mask().nonzero()[0].tolist()
        # legal_action_list = [_ for _ in range(self.env.action_space.n)]
        return legal_action_list

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)

        self.rewards.append(reward)
        self.history.append(action)
        self.obs_history.append(obs)

        return self.obs(len(self.rewards)), reward, done, _

    def reset(self, train=False):
        obs = self.env.reset()

        self.rewards = []
        self.history = []
        self.obs_history = []

        for _ in range(self.k):
            self.obs_history.append(obs)

        return self.obs(0)

    def close(self):
        self.env.close()

    def obs(self, i):
        frames = self.obs_history[i:i + self.k]

        return np.array(frames).flatten()

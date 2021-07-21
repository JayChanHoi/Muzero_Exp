from collections import deque

import numpy as np

import torch

from ...core.game import Game

class ClassicControlWrapper(Game):
    def __init__(self, env, k: int, discount: float):
        """

        :param env: instance of gym environment
        :param k: no. of observations to stack
        """
        super().__init__(env, env.action_space.n, discount, k=k)
        self.k = k
        self.frames = deque([], maxlen=k)

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.rewards.append(reward)
        self.history.append(action)
        self.obs_history.append(obs)

        return self.obs(len(self.rewards)), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

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
        # obs = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
        return np.array(frames).flatten()

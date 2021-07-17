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

        return self.obs(len(self.rewards)), reward, done, _

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
        job_raw_feature_list = []
        plant_raw_feature_list = []
        crew_raw_feature_list = []
        misc_info_raw_feature_list = []
        for item in self.obs_history[i:i + self.k]:
            job_raw_feature_list.append(item[0])
            plant_raw_feature_list.append(item[1])
            crew_raw_feature_list.append(item[2])
            misc_info_raw_feature_list.append(item[3])
        job_raw_feature = np.stack(job_raw_feature_list, axis=0)
        plant_raw_feature = np.stack(plant_raw_feature_list, axis=0)
        crew_raw_feature = np.stack(crew_raw_feature_list, axis=0)
        misc_info_raw_feature = np.stack(misc_info_raw_feature_list, axis=0)

        return tuple([job_raw_feature, plant_raw_feature, crew_raw_feature, misc_info_raw_feature])

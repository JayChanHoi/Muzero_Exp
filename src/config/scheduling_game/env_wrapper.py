from collections import deque

import numpy as np

# from ...core.game import Game, Action
from ...core.game import Game

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
        self.job_feature_dim, self.plant_feature_dim, self.crew_feature_dim, self.misc_info_feature_dim, self.job_num, \
        self.plant_num, self.truck_num, _ = self.get_obs_info()

    def legal_actions(self):
        legal_action_list = [0] + self.env.get_available_actions().flatten().nonzero().squeeze().tolist()
        # return [Action(_) for _ in legal_action_list]
        return legal_action_list

    def get_obs_info(self):
        return self.env.get_obs_info()

    def convert_action(self, action, plant_num, truck_num):
        if action == 0:
            if_iterate = 1
            selected_job = -1
            selected_plant = -1
            selected_crew = -1
        else:
            if_iterate = 0
            selected_job = (action - 1) // (plant_num * truck_num)
            selected_plant = (action - 1 - selected_job * (plant_num * truck_num)) // truck_num
            selected_crew = action - 1 - selected_job * (plant_num * truck_num) - selected_plant * truck_num

        return selected_job, selected_plant, selected_crew, if_iterate

    def step(self, action):
        selected_job, selected_plant, selected_crew, if_iterate = self.convert_action(action, self.plant_num, self.truck_num)
        obs, reward, done, _ = self.env.step((if_iterate, selected_job, selected_plant, selected_crew))

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

import torch
import torch.nn as nn

from ...core.model import BaseMuZeroNet

class StateEncoder(nn.Module):
    def __init__(self,
                 dropout_p,
                 job_feature_dim,
                 plant_feature_dim,
                 crew_feature_dim,
                 misc_info_feature_dim,
                 plant_num,
                 truck_num,
                 job_num):
        super(StateEncoder, self).__init__()
        self.job_encoder = nn.Sequential(
            nn.Linear(job_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.plant_encoder = nn.Sequential(
            nn.Linear(plant_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.crew_encoder = nn.Sequential(
            nn.Linear(crew_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.misc_encoder = nn.Sequential(
            nn.Linear(misc_info_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.job_aggregator = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.plant_aggregator = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.crew_aggregator = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.misc_aggregator = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.job_num = job_num
        self.plant_num = plant_num
        self.truck_num = truck_num

    def _single_time_forward(self, single_time_input):
        job_raw_feature, plant_raw_feature, crew_raw_feature, misc_info_raw_feature = single_time_input

        print('test!', misc_info_raw_feature.shape)

        encoded_job_rep = self.job_encoder(job_raw_feature.reshape(job_raw_feature.shape[0]*job_raw_feature.shape[1], -1)).view(job_raw_feature.shape[0], job_raw_feature.shape[1], -1)
        encoded_plant_rep = self.plant_encoder(plant_raw_feature.reshape(plant_raw_feature.shape[0]*plant_raw_feature.shape[1], -1)).view(plant_raw_feature.shape[0], plant_raw_feature.shape[1], -1)
        encoded_crew_rep = self.crew_encoder(crew_raw_feature.reshape(crew_raw_feature.shape[0]*crew_raw_feature.shape[1], -1)).view(crew_raw_feature.shape[0], crew_raw_feature.shape[1], -1)
        encoded_misc_rep = self.misc_encoder(misc_info_raw_feature)

        aggregated_encoded_job_rep = self.job_aggregator(encoded_job_rep.mean(dim=1))
        aggregated_encoded_plant_rep = self.plant_aggregator(encoded_plant_rep.mean(dim=1))
        aggregated_encoded_crew_rep = self.crew_aggregator(encoded_crew_rep.mean(dim=1))
        aggregated_encoded_misc_rep = self.misc_aggregator(encoded_misc_rep)
        global_rep = self.global_encoder(aggregated_encoded_job_rep+aggregated_encoded_plant_rep+aggregated_encoded_crew_rep+aggregated_encoded_misc_rep)

        encoded_state_rep = torch.cat(
            [
                encoded_job_rep.view(encoded_job_rep.shape[0], -1),
                encoded_plant_rep.view(encoded_plant_rep.shape[0], -1),
                encoded_crew_rep.view(encoded_crew_rep.shape[0], -1),
                global_rep
            ],
            dim=1
        )

        return encoded_state_rep

    def forward(self, input):
        job_raw_feature, plant_raw_feature, crew_raw_feature, misc_info_raw_feature = input

        encoded_state_list = []
        for i in range(job_raw_feature.shape[1]):
            state_encoder_input = [
                job_raw_feature[:, i, :, :],
                plant_raw_feature[:, i, :, :],
                crew_raw_feature[:, i, :, :],
                misc_info_raw_feature[:, i, :]
            ]
            encoded_state_list.append(self._single_time_forward(state_encoder_input))
        aggregated_encoded_state = torch.cat(encoded_state_list, dim=1)

        return aggregated_encoded_state

class MuZeroNetConcreteSchedulingGame(BaseMuZeroNet):
    def __init__(self,
                 job_feature_dim,
                 plant_feature_dim,
                 crew_feature_dim,
                 misc_info_feature_dim,
                 plant_num,
                 truck_num,
                 job_num,
                 action_space_n,
                 reward_support_size,
                 value_support_size,
                 inverse_value_transform,
                 inverse_reward_transform):
        super(MuZeroNetConcreteSchedulingGame, self).__init__(inverse_value_transform, inverse_reward_transform)
        self.hx_size = 256
        self._representation = StateEncoder(
            dropout_p=0.1,
            job_feature_dim=job_feature_dim,
            plant_feature_dim=plant_feature_dim,
            crew_feature_dim=crew_feature_dim,
            misc_info_feature_dim=misc_info_feature_dim,
            plant_num=plant_num,
            truck_num=truck_num,
            job_num=job_num
        )
        # self._representation = nn.Sequential(
        #     nn.Linear(input_size, self.hx_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.hx_size, self.hx_size),
        #     nn.ReLU(),
        # )
        self._dynamics_state = nn.Sequential(
            nn.Linear(self.hx_size + action_space_n, self.hx_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hx_size * 2, self.hx_size),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(self.hx_size, self.hx_size),
            nn.ReLU(),
        )
        self._dynamics_reward = nn.Sequential(
            nn.Linear(self.hx_size + action_space_n, self.hx_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hx_size * 2, self.hx_size * 2),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(self.hx_size * 2, reward_support_size)
        )
        self._prediction_actor = nn.Sequential(
            nn.Linear(self.hx_size, self.hx_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hx_size * 2, self.hx_size * 2),
            nn.LeakyReLU(),
            nn.Linear(self.hx_size * 2, action_space_n)
        )
        self._prediction_value = nn.Sequential(
            nn.Linear(self.hx_size, self.hx_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hx_size * 2, self.hx_size * 2),
            nn.LeakyReLU(),
            nn.Linear(self.hx_size * 2, value_support_size)
        )
        self.action_space_n = action_space_n

        self._prediction_value[-1].weight.data.fill_(0)
        self._prediction_value[-1].bias.data.fill_(0)
        self._dynamics_reward[-1].weight.data.fill_(0)
        self._dynamics_reward[-1].bias.data.fill_(0)

    def prediction(self, state):
        actor_logit = self._prediction_actor(state)
        value = self._prediction_value(state)
        return actor_logit, value

    def representation(self, obs_history):
        return self._representation(obs_history)

    def dynamics(self, state, action):
        assert len(state.shape) == 2
        assert action.shape[1] == 1

        action_one_hot = torch.zeros(size=(action.shape[0], self.action_space_n),
                                     dtype=torch.float32, device=action.device)
        action_one_hot.scatter_(1, action, 1.0)

        x = torch.cat((state, action_one_hot), dim=1)
        next_state = self._dynamics_state(x)
        reward = self._dynamics_reward(x)
        return next_state, reward

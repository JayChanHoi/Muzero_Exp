import torch
import torch.nn as nn

from ...core.model import BaseMuZeroNet

class ResidualMLPBlock(nn.Module):
    def __init__(self, dropout_p):
        super(ResidualMLPBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.block(x)
        out += identity
        out = self.relu(out)

        return out

class StateEncoder(nn.Module):
    def __init__(self, dropout_p):
        super(StateEncoder, self).__init__()
        self.fc_1 = nn.Linear(488, 256)
        self.relu_1 = nn.ReLU(inplace=True)

        self.residual_blocks = nn.ModuleList([ResidualMLPBlock(dropout_p) for _ in range(2)])

        self.fc_last = nn.Linear(256,512)
        self.relu_last = nn.ReLU(inplace=True)

    def forward(self, x):
        x_ = self.fc_1(x)
        x_ = self.relu_1(x_)

        for block in self.residual_blocks:
            x_ = block(x_)

        x_ = self.fc_last(x_)
        x_ = self.relu_last(x_)

        return x_

class BitcoinTradeNet(BaseMuZeroNet):
    def __init__(self,
                 action_space_n,
                 reward_support_size,
                 value_support_size,
                 inverse_value_transform,
                 inverse_reward_transform):
        super(BitcoinTradeNet, self).__init__(inverse_value_transform, inverse_reward_transform)
        self.hx_size = 512
        self._representation = StateEncoder(
            dropout_p=0.1
        )

        self._dynamics_state = nn.Sequential(
            nn.Linear(self.hx_size + action_space_n, self.hx_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hx_size * 2, self.hx_size),
            nn.ReLU(),
            nn.Linear(self.hx_size, self.hx_size),
            nn.ReLU(),
        )
        self._dynamics_reward = nn.Sequential(
            nn.Linear(self.hx_size + action_space_n, self.hx_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hx_size * 2, self.hx_size * 2),
            nn.LeakyReLU(),
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

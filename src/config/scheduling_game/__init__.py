import gym
import torch
from ...core.config import BaseMuZeroConfig, DiscreteSupport
from .env_wrapper import SchedulingGameWrapper
from .model import MuZeroNetConcreteSchedulingGame
from ...environment.concrete_production_delivery_env import ConcreteProductionDeliveryEnvV2

class ScheudlingControlConfig(BaseMuZeroConfig):
    def __init__(self, env_config_path):
        super(ScheudlingControlConfig, self).__init__(
            training_steps=20000,
            test_interval=100,
            test_episodes=5,
            checkpoint_interval=20,
            max_moves=100,
            discount=0.997,
            dirichlet_alpha=0.25,
            num_simulations=25,
            batch_size=128,
            td_steps=5,
            num_actors=16,
            lr_init=0.05,
            lr_decay_rate=0.01,
            lr_decay_steps=10000,
            window_size=10000,
            value_loss_coeff=1,
            value_support=DiscreteSupport(-20, 20),
            reward_support=DiscreteSupport(-5, 5),
        )
        self.env_config_path = env_config_path

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        game = self.new_game()
        self.job_feature_dim, self.plant_feature_dim, self.crew_feature_dim, self.misc_info_feature_dim, self.plant_num, \
        self.truck_num, self.job_num, _ = game.get_obs_info()
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        return MuZeroNetConcreteSchedulingGame(
            self.job_feature_dim,
            self.plant_feature_dim,
            self.crew_feature_dim,
            self.misc_info_feature_dim,
            self.plant_num,
            self.truck_num,
            self.job_num,
            self.action_space_size,
            self.reward_support.size,
            self.value_support.size,
            self.inverse_value_transform,
            self.inverse_reward_transform
        )

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None):
        env = ConcreteProductionDeliveryEnvV2(self.env_config_path)

        return SchedulingGameWrapper(env, discount=self.discount, k=4)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)


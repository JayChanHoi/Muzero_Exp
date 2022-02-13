from .utils import read_bitcoin_history
import numpy as np
import random

import gym
from gym import spaces

class BitcoinTradeEnv(gym.Env):
    def __init__(self, env_config):
        self.trading_records = read_bitcoin_history(os.path.join(os.path.dirname(__file__), 'Binance_BTCUSDT_minute.csv'))
        self.trading_open_price = self.trading_records[:, 0]
        self.trading_close_price = self.trading_records[:, -3]
        self.env_config = env_config
        self.action_space = spaces.Discrete(11)

    def _get_trade_rep(self, trading_index):
        norm_constant = np.array([5e4, 5e4, 5e6]).reshape(1, -1)
        open = self.trading_records[trading_index-39:trading_index+1, 0].reshape(-1, 1)
        close = self.trading_records[trading_index-39:trading_index+1, 3].reshape(-1, 1)
        volume = self.trading_records[trading_index-39:trading_index+1, -1].reshape(-1, 1)
        trade_rep = np.concatenate([open, close, volume], axis=1)

        return (trade_rep / norm_constant).reshape(-1)

    def _prep_obs(self):
        '''
        the obs is not yet normalized.
        :return:
        '''
        raw_rep = self._get_trade_rep(self.trading_index)
        obs = np.concatenate([raw_rep, np.array([self.current_cash_value/self.env_config.initial_cash_value, (self.current_asset_unit * self.trading_open_price[self.trading_index+1])/self.env_config.initial_cash_value])], axis=0)

        return obs

    def get_action_mask(self):
        mask_rep = [
            1,
            self.current_cash_value - self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 1 - self.env_config.trade_cost,
            self.current_cash_value - self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 2 - self.env_config.trade_cost,
            self.current_cash_value - self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 3 - self.env_config.trade_cost,
            self.current_cash_value - self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 4 - self.env_config.trade_cost,
            self.current_cash_value - self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 5 - self.env_config.trade_cost,
            (self.current_cash_value - self.env_config.trade_cost) + (self.current_asset_unit * self.trading_open_price[self.trading_index+1]) -  self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 1,
            (self.current_cash_value - self.env_config.trade_cost) + (self.current_asset_unit * self.trading_open_price[self.trading_index+1]) -  self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 2,
            (self.current_cash_value - self.env_config.trade_cost) + (self.current_asset_unit * self.trading_open_price[self.trading_index+1]) -  self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 3,
            (self.current_cash_value - self.env_config.trade_cost) + (self.current_asset_unit * self.trading_open_price[self.trading_index+1]) -  self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 4,
            (self.current_cash_value - self.env_config.trade_cost) + (self.current_asset_unit * self.trading_open_price[self.trading_index+1]) -  self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 5,
        ]
        mask =  np.greater_equal(np.array(mask_rep), 0)

        return mask

    def reset(self, train:bool):
        self.act_count = 0
        self.current_cash_value = self.env_config.initial_cash_value
        self.current_asset_unit = 0
        self.asset_cash_history = [[self.current_cash_value, self.current_asset_unit]]
        self.total_capital_history = [self.current_cash_value]
        if train:
            self.trading_index = random.randint(60+3, self.trading_records.shape[0] - self.env_config.episode_length - 20000)
        else:
            self.trading_index = random.randint(self.trading_records.shape[0] - 20000, self.trading_records.shape[0] - self.env_config.episode_length)
            
        self.obs = self._prep_obs()

        return self.obs

    def reset_but_not_sampling_new_trading_index(self):
        self.act_count = 0
        self.current_cash_value = self.env_config.initial_cash_value
        self.current_asset_unit = 0
        self.asset_cash_history = [[self.current_cash_value, self.current_asset_unit]]
        self.total_capital_history = [self.current_cash_value]
        self.trading_index -= self.env_config.episode_length
        self.obs = self._prep_obs()

        return self.obs

    def _act(self, action):
        '''
        assuming each time of buying or selling position are having constant amount and only have long position.
        :param action: int -> only have three choices: (0, 1, 2). 0 -> hold, 1 -> buy, 2 -> sell
        :return: reward and done
        '''
        done = False

        if action == 0:
            pass
        elif action == 1:
            self.current_asset_unit += self.env_config.position_amount
            self.current_cash_value -= (self.trading_open_price[self.trading_index+1] * self.env_config.position_amount + self.env_config.trade_cost)
        elif action == 2:
            self.current_asset_unit += self.env_config.position_amount * 2
            self.current_cash_value -= (self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 2 + self.env_config.trade_cost)
        elif action == 3:
            self.current_asset_unit += self.env_config.position_amount * 3
            self.current_cash_value -= (self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 3 + self.env_config.trade_cost)
        elif action == 4:
            self.current_asset_unit += self.env_config.position_amount * 4
            self.current_cash_value -= (self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 4 + self.env_config.trade_cost)
        elif action == 5:
            self.current_asset_unit += self.env_config.position_amount * 5
            self.current_cash_value -= (self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 5 + self.env_config.trade_cost)
        elif action == 6:
            self.current_asset_unit -= self.env_config.position_amount
            self.current_cash_value += (self.trading_open_price[self.trading_index+1] * self.env_config.position_amount - self.env_config.trade_cost)
        elif action == 7:
            self.current_asset_unit -= self.env_config.position_amount * 2
            self.current_cash_value += (self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 2 - self.env_config.trade_cost)
        elif action == 8:
            self.current_asset_unit -= self.env_config.position_amount * 3
            self.current_cash_value += (self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 3 - self.env_config.trade_cost)
        elif action == 9:
            self.current_asset_unit -= self.env_config.position_amount * 4
            self.current_cash_value += (self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 4 - self.env_config.trade_cost)
        elif action == 10:
            self.current_asset_unit -= self.env_config.position_amount * 5
            self.current_cash_value += (self.trading_open_price[self.trading_index+1] * self.env_config.position_amount * 5 - self.env_config.trade_cost)
        else:
            raise ValueError('action should only be picked from 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10')

        reward = (self.current_cash_value + self.current_asset_unit*self.trading_open_price[self.trading_index+1] - self.total_capital_history[-1]) / 10000

        self.asset_cash_history.append([self.current_cash_value, self.current_asset_unit])
        self.total_capital_history.append(self.current_cash_value + self.current_asset_unit*self.trading_open_price[self.trading_index+1])
        self.act_count += 1
        self.trading_index += 1
        if self.act_count == self.env_config.episode_length:
            done = True
            reward += (self.current_cash_value + self.current_asset_unit*self.trading_open_price[self.trading_index+1] - self.total_capital_history[0]) / 10000

        return reward, done

    def get_action_number(self):
        return self.get_action_mask().shape[0]

    def step(self, action):
        '''
        step function should return next obs and reward
        :param action: action
        :return: next_obs, reward, done, extra_info
        '''
        reward, done = self._act(action)
        self.obs = self._prep_obs()

        return self.obs, reward, done, None

    def render(self):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    import os
    import yaml
    from collections import namedtuple
    from itertools import count

    config_path = os.path.join(os.getcwd(), 'bitcoin/env_config.yml')
    env_config = yaml.load(open(config_path, 'r'))
    env_config = namedtuple('env_config', env_config.keys())(**env_config)
    bitcon_trade_env = BitcoinTradeEnv(env_config)
    state = bitcon_trade_env.reset()
    rewards = []
    for _ in count():
        action = np.random.randint(0, 3)
        state, reward, done, _ = bitcon_trade_env.step(action)
        print((1- bitcon_trade_env.get_action_mask()))
        rewards.append(reward)
        print(action)
        print(state)
        print(reward)
        print(done)
        print(bitcon_trade_env.total_capital_history[-1])
        print(state.shape)
        print('-------------------------------------------------------------------------------')
        if done:
            break
    print(np.mean(rewards))
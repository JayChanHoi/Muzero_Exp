import torch

import numpy as np

import gym
from gym import spaces

import yaml
from datetime import datetime, timedelta
import random
import math

from .auxilary import Crew, Job, Plant
from .utils import data_injection_handler

class ConcreteProductionDeliveryEnv(gym.Env):
    def __init__(self, env_config_path, device='cpu'):
        assert env_config_path is not None, 'env_config_path have to be provided'
        self.env_config = yaml.load(open(env_config_path, "r"))
        self.device = device
        self.plant = Plant(
            self.env_config["plant_dict"],
            device
        )
        self.crew = Crew(
            self.env_config["crew_preserved_capacity"],
            self.env_config["max_cap_per_truck"],
            self.env_config["start_time_str"],
            device
        )
        self.job = Job(
            self.env_config["job_dict"],
            self.env_config["max_cap_per_truck"],
            device
        )
        self.start_time = datetime.strptime(self.env_config["start_time_str"], "%H:%M:%S")
        self.time_step_size = self.env_config["time_step_size"]
        self.plant_process_duration = self.env_config["plant_process_time"]
        self.job_process_duration = self.env_config["job_process_time"]
        self.truck_travel_speed = self.env_config["truck_travel_speed"]
        self.coor_id_tensor = torch.tensor(list(self.env_config["coor_id_dict"].values())).float().to(self.device)
        self.coor_tensor_job, self.coor_tensor_plant = self._get_coor_tensor_job_plant()
        self.job_num = self.job.job_capacity_state_dict.__len__()
        self.plant_num = self.plant.plant_capacity_state_dict.__len__()
        self.truck_num = len(self.crew.crew_id)
        self.benchmark_truck_in_garage = [self.env_config["crew_preserved_capacity"] for i in range(10)]
        self.reset()

    def _get_coor_tensor_job_plant(self):
        job_coor_list = []
        for job_id in sorted(self.job.job_dict.keys()):
            job_coor_list.append(self.coor_id_tensor[self.job.job_dict[job_id][0]])

        plant_coor_list = []
        for plant_id in sorted(self.plant.plant_dict.keys()):
            plant_coor_list.append(self.coor_id_tensor[self.plant.plant_dict[plant_id][0]])

        job_coor_tensor = torch.stack(job_coor_list, dim=0)
        plant_coor_tensor = torch.stack(plant_coor_list, dim=0)

        return job_coor_tensor, plant_coor_tensor

    def _get_distance(self, a, b):
        """
        :param A: source coor tensor -> (n, 2)
        :param B: reference coor tensor -> (m, 2)
        :return: distance tensor (n, m)
        """
        a_broadcast = a.unsqueeze(1).repeat(1, b.shape[0], 1)
        b_broadcast = b.unsqueeze(0).repeat(a.shape[0], 1, 1)
        distance = torch.norm(a_broadcast - b_broadcast, p=2, dim=2)

        return distance

    def _normalize(self, distance):
        """
        The distance here are calculated straightly by using shortest distance in 2d Euclidean space. In general, it can
        be provided with a route distance map form place to place.
        :return: tensor in shape (n, m)
        """
        min_distance_on_b = distance.min(dim=1, keepdims=True)[0]
        max_min_diff_distance_on_b = distance.max(dim=1, keepdims=True)[0] - distance.min(dim=1, keepdims=True)[0]
        normalized_distance = (distance - min_distance_on_b) / (max_min_diff_distance_on_b + 10e-5)

        return normalized_distance

    def _check_if_available(self):
        """
        True if there is available allocation of resources under constraints.
        :return:
        """
        base_mask = self.crew.crew_state_tensor.view(1, 1, -1).repeat(self.job_num, self.plant_num, 1)

        # plant task
        plant_truck_mask_list = []
        plant_task_duration = self.plant_process_duration
        for plant_id in self.plant.plant_capacity_state_dict.keys():
            plant_resource_rep = self.plant.plant_capacity_state_dict[plant_id]
            task_start_time = ((self.current_time - self.start_time).seconds / 60 + (self.plant_truck_distance[plant_id] / self.truck_travel_speed) * 60).long()
            task_end_time = (task_start_time + plant_task_duration).long()
            task_time = torch.stack([task_start_time, task_end_time], dim=1)
            task_time_plant_resource_overlap = (task_time.flip(dims=[1]).view(self.truck_num, 1, 1, 2).float()  - plant_resource_rep.unsqueeze(0)).sign()
            truck_plant_resource_available = ((task_time_plant_resource_overlap.prod(dim=-1).eq(-1)).sum(-1) == 0).float()
            plant_truck_mask_list.append((truck_plant_resource_available.sum(-1) != 0).float())

        # shape -> (m, k)
        plant_truck_mask = torch.stack(plant_truck_mask_list, dim=0).unsqueeze(0).repeat(self.job_num, 1, 1)

        # job_task
        job_truck_mask_list = []
        job_task_duration = self.job_process_duration
        for job_id in self.job.job_capacity_state_dict.keys():
            job_resource_rep = self.job.job_capacity_state_dict[job_id]
            distance = self.job_plant_distance[job_id].unsqueeze(1).repeat(1, self.truck_num) + self.plant_truck_distance
            task_start_time = ((self.current_time - self.start_time).seconds / 60 + (distance / self.truck_travel_speed) * 60 + plant_task_duration).long()
            task_end_time = (task_start_time + job_task_duration).long()
            task_time = torch.stack([task_start_time, task_end_time], dim=2)
            task_time_job_resource_overlap = (task_time.flip(dims=[2]).view(self.plant_num, self.truck_num, 1, 1, 2).float() - job_resource_rep.unsqueeze(0).unsqueeze(0)).sign()
            truck_job_resource_available = ((task_time_job_resource_overlap.prod(dim=-1).eq(-1)).sum(-1) == 0).float()
            job_truck_mask_list.append((truck_job_resource_available.sum(-1) != 0).float())

        # shape -> (n, m, k)

        job_truck_mask = (torch.stack(job_truck_mask_list, dim=0) * (self.job.job_order_qty_tensor > 0).float().view(-1, 1, 1))

        return (((base_mask * job_truck_mask * plant_truck_mask).view(-1).sum()).item() != 0)

    def _iterate(self):
        """
        when there is no more actions available, will iterate along time flow and resource and production capacity will release along with time.
        :return:
        """
        self.crew.update(self.time_step_size)
        self.current_time += timedelta(minutes=self.time_step_size)

    def _get_temp_obs(self):
        """
        if there is no more action available after the action is taken, will iterate through time flow until there is action availble and
        return the obs when there is action available again.
        :param obs:
        :param action:
        :param reward:
        :return:
        """
        normalized_distance_job_to_truck = self._normalize(self.job_truck_distance)
        normalized_distance_plant_to_truck = self._normalize(self.plant_truck_distance)
        available_job_plant_truck = self.get_available_actions()

        job_rep_list = []
        for job_id in self.job.job_dict.keys():
            job_resource_rep = self.job.job_capacity_state_dict[job_id]
            job_resource_capacity_remain = (1 - (job_resource_rep[:, :, 1] - job_resource_rep[:, :, 0]).sum(-1) / (600)).mean().unsqueeze(0)
            job_order_qty_remain = self.job.job_order_qty_tensor[job_id]
            job_order_remain = (job_order_qty_remain / (self.job.job_dict[job_id][2] + 0.01)).to(self.device)
            normalized_distance_to_plant = self.normalized_distance_job_to_plant[job_id, :]
            normalized_distance_to_truck = normalized_distance_job_to_truck[job_id, :]
            process_duration = torch.tensor([self.job_process_duration / 60]).to(self.device)
            job_rep = torch.cat(
                [
                    job_resource_capacity_remain,
                    job_order_remain.unsqueeze(0),
                    normalized_distance_to_plant,
                    normalized_distance_to_truck,
                    process_duration
                ],
                dim=0
            )
            job_rep_list.append(job_rep)
        job_reps = torch.stack(job_rep_list, dim=0)

        plant_rep_list = []
        for plant_id in self.plant.plant_dict.keys():
            plant_resource_rep = self.plant.plant_capacity_state_dict[plant_id]
            plant_resource_capacity_remain = (1 - (plant_resource_rep[:, :, 1] - plant_resource_rep[:, :, 0]).sum(-1) / (600)).mean().unsqueeze(0)
            normalized_distance_to_job = self.normalized_distance_job_to_plant[:, plant_id]
            normalized_distance_to_truck = normalized_distance_plant_to_truck[plant_id, :]
            process_duration = torch.tensor([self.plant_process_duration / 60]).to(self.device)
            plant_rep = torch.cat(
                [
                    plant_resource_capacity_remain,
                    normalized_distance_to_job,
                    normalized_distance_to_truck,
                    process_duration
                ],
                dim=0
            )
            plant_rep_list.append(plant_rep)
        plant_reps = torch.stack(plant_rep_list, dim=0)

        crew_features = torch.cat([normalized_distance_job_to_truck.t(), normalized_distance_plant_to_truck.t()], dim=1)
        crew_rep_list = []
        for truck_id in self.crew.crew_id:
            truck_available_action = available_job_plant_truck[:, :, truck_id].reshape(-1)
            travel_speed = torch.tensor([self.truck_travel_speed]).float().to(self.device) / 1000
            truck_rep = torch.cat(
                [
                    truck_available_action,
                    crew_features[truck_id, :],
                    travel_speed
                ],
                dim=0
            )
            crew_rep_list.append(
                truck_rep
            )
        crew_reps = torch.stack(crew_rep_list, dim=0)
        available_action = available_job_plant_truck.view(-1)
        current_work_duration, standard_work_duration = self._get_work_duration()
        work_duration = current_work_duration / standard_work_duration
        order_remain = (self.job.job_order_qty_tensor.sum() / self.job.job_initial_order_qty).to(self.device)
        miscellaneous_rep = torch.cat(
            [
                available_action,
                torch.tensor([work_duration], dtype=torch.float).to(self.device),
                order_remain.unsqueeze(0),
                self.crew.crew_state_tensor.to(self.device)
            ],
            dim=0
        )

        temp_obs = tuple([job_reps, plant_reps, crew_reps, miscellaneous_rep])

        return temp_obs

    def _get_observations(self):
        job_reps_hist = []
        plant_reps_hist = []
        crew_reps_hist = []
        misc_reps_hist = []
        for obs in self.obs_hist:
            job_reps_hist.append(obs[0])
            plant_reps_hist.append(obs[1])
            crew_reps_hist.append(obs[2])
            misc_reps_hist.append(obs[3])

        observation = tuple(
            [
                torch.stack(job_reps_hist, dim=0),
                torch.stack(plant_reps_hist, dim=0),
                torch.stack(crew_reps_hist, dim=0),
                torch.stack(misc_reps_hist, dim=0),
            ])

        return observation

    def _get_work_duration(self):
        current_work_duration = (self.current_time - datetime.strptime(self.env_config["start_time_str"], "%H:%M:%S")).seconds / 60
        standard_work_duration = (datetime.strptime(self.env_config["off_time_str"], "%H:%M:%S") - datetime.strptime(self.env_config["start_time_str"], "%H:%M:%S")).seconds / 60

        return current_work_duration, standard_work_duration

    def _act(self, action):
        """
        task duration need to compute by using the coor id and production time. action space will only include the available actions can be made.
        The distance here are calculated straightly by using shortest distance in 2d Euclidean space. In general, it can
        be provided with a route distance map form place to place.
        :param observation:
        :param action: int -> action id
        :return: reward
        """
        if action[0] == 0:
            selected_job, selected_plant, selected_truck = action[1:]
            if self.get_available_actions()[selected_job, selected_plant, selected_truck].item() == 0:
                self._iterate()
                reward = -1
            else:
                if self.crew.crew_coor_tensor[selected_truck, 0].item() == -1:
                    plant_truck_dist = 0
                else:
                    plant_truck_dist = torch.norm(self.crew.crew_coor_tensor[selected_truck] - self.coor_id_tensor[selected_plant], p=2)
                plant_job_dist = torch.norm(self.coor_tensor_plant[selected_plant] - self.coor_tensor_job[selected_job], p=2)

                # execute truck
                truck_task_duration = int(((plant_job_dist + plant_truck_dist) / self.truck_travel_speed) * 60 + self.plant_process_duration  + self.job_process_duration)
                self.crew.execute(selected_truck, truck_task_duration, self.coor_tensor_job[selected_job])

                # execute plant
                plant_task_duration = self.plant_process_duration
                plant_task_start_time = int((self.current_time - self.start_time).seconds / 60 + (plant_truck_dist / self.truck_travel_speed) * 60)
                self.plant.execute(selected_plant, plant_task_duration, plant_task_start_time)

                # execute job
                job_task_duration = self.job_process_duration
                job_task_start_time = int((self.current_time - self.start_time).seconds / 60 + ((plant_job_dist + plant_truck_dist) / self.truck_travel_speed) * 60 + self.plant_process_duration)
                self.job.execute(selected_job, job_task_duration, job_task_start_time)

                # update job_resource and plant_resource distance tensor
                truck_coor_tensor = self.crew.crew_coor_tensor.clone().to(self.device)
                truck_coor_tensor[selected_truck, :] = self.coor_id_tensor[self.job.job_dict[selected_job][0]]
                mask = truck_coor_tensor[:, 0].ne(-1)
                truck_coor_tensor *= mask.view(-1, 1).repeat(1, 2)
                job_truck_distance = self._get_distance(self.coor_tensor_job, truck_coor_tensor) * (mask.view(1, -1).repeat(self.coor_tensor_job.shape[0], 1))
                plant_truck_distance = self._get_distance(self.coor_tensor_plant, truck_coor_tensor) * (mask.view(1, -1).repeat(self.coor_tensor_plant.shape[0], 1))
                self.job_truck_distance = job_truck_distance
                self.plant_truck_distance = plant_truck_distance

                reward = (self.crew.number_of_trucks_in_garage().item() / self.env_config['crew_preserved_capacity']) * 1.0

        elif action[0] == 1:
            self._iterate()
            reward = 0

        else:
            raise ValueError('action[0] should be either 0 or 1')

        self.number_of_trucks_in_garage.append(self.crew.number_of_trucks_in_garage().item())
        self.execute_count += 1

        if self.job.job_order_qty_tensor.sum() == 0:
            done = True
        else:
            done = False

        if not done:
            if self.current_time + timedelta(minutes=self.crew.crew_task_duration_tensor.max(dim=0)[0].item()) >= datetime.strptime(self.env_config['max_time_str'], "%H:%M:%S"):
                done = True

            while not self._check_if_available():
                self._iterate()
                if self.current_time + timedelta(minutes=self.crew.crew_task_duration_tensor.max(dim=0)[0].item()) >= datetime.strptime(self.env_config['max_time_str'], "%H:%M:%S"):
                    done = True

        if done:
            if self.current_time + timedelta(minutes=self.crew.crew_task_duration_tensor.max(dim=0)[0].item()) < datetime.strptime(self.env_config['max_time_str'], "%H:%M:%S"):
                number_of_trucks_on_road = self.env_config['crew_preserved_capacity'] -  min(self.number_of_trucks_in_garage) + 0.0001
                order_qty_delivered = (self.job.job_initial_order_qty.sum() - self.job.job_order_qty_tensor.sum()).item()
                truck_on_road_productivity = (order_qty_delivered / self.env_config['max_cap_per_truck']) / number_of_trucks_on_road
                reward += truck_on_road_productivity / 1
            else:
                reward -= 2

        return reward, done

    def reset_job_order_distribution(self):
        job_dict = self.env_config["job_dict"].copy()
        order_qty = random.randint(2000, 3500)
        sampled_order_qty_list = (torch.rand(job_dict.keys().__len__()).softmax(dim=0) * order_qty).int().tolist()
        for index, job_id in enumerate(job_dict.keys()):
            job_order_qty = sampled_order_qty_list[index]
            job_dict[job_id][2] = job_order_qty

        self.job = Job(
            job_dict,
            self.env_config["max_cap_per_truck"],
            self.device
        )
        observation = self.reset()

        return observation

    def get_eval_info(self):
        current_work_duration, standard_work_duration = self._get_work_duration()
        ot_duration = current_work_duration + self.crew.crew_task_duration_tensor.max(dim=0)[0].item() - standard_work_duration

        max_number_of_trucks_on_road = self.env_config["crew_preserved_capacity"] - min(self.number_of_trucks_in_garage)

        qty_remain_ratio = (self.job.job_order_qty_tensor.sum() / self.job.job_initial_order_qty.sum()).item()

        number_of_trucks_on_road = self.env_config['crew_preserved_capacity'] -  min(self.number_of_trucks_in_garage) + 0.0001
        order_qty_delivered = (self.job.job_initial_order_qty.sum() - self.job.job_order_qty_tensor.sum()).item()
        truck_on_road_productivity = (order_qty_delivered / self.env_config['max_cap_per_truck']) / number_of_trucks_on_road

        return max_number_of_trucks_on_road, ot_duration, qty_remain_ratio, truck_on_road_productivity

    def get_obs_info(self):
        job_dim = self.observation[0].shape[2]
        plant_dim = self.observation[1].shape[2]
        crew_dim = self.observation[2].shape[2]
        misc_info_dim = self.observation[3].shape[1]
        hist_length = self.observation[0].shape[0]

        return job_dim, plant_dim, crew_dim, misc_info_dim, self.job_num, self.plant_num, self.truck_num, hist_length

    def get_task_duration(self, selected_job, selected_plant, selected_truck):
        if self.crew.crew_coor_tensor[selected_truck, 0].item() == -1:
            plant_truck_dist = 0
        else:
            plant_truck_dist = torch.norm(self.crew.crew_coor_tensor[selected_truck] - self.coor_id_tensor[selected_plant], p=2)
        plant_job_dist = torch.norm(self.coor_id_tensor[selected_plant] - self.coor_id_tensor[selected_job], p=2)
        task_duration = ((plant_job_dist + plant_truck_dist) / self.truck_travel_speed)*60 + self.plant_process_duration  + self.job_process_duration

        return task_duration

    def get_available_actions(self):
        base_mask = self.crew.crew_state_tensor.view(1, 1, -1).repeat(self.job_num, self.plant_num, 1)

        # plant task
        plant_truck_mask_list = []
        plant_task_duration = self.plant_process_duration
        for plant_id in self.plant.plant_capacity_state_dict.keys():
            plant_resource_rep = self.plant.plant_capacity_state_dict[plant_id]
            task_start_time = ((self.current_time - self.start_time).seconds / 60 + (self.plant_truck_distance[plant_id] / self.truck_travel_speed) * 60).long()
            task_end_time = (task_start_time + plant_task_duration).long()
            task_time = torch.stack([task_start_time, task_end_time], dim=1)
            task_time_plant_resource_overlap = (task_time.flip(dims=[1]).view(self.truck_num, 1, 1, 2).float()  - plant_resource_rep.unsqueeze(0)).sign()
            truck_plant_resource_available = ((task_time_plant_resource_overlap.prod(dim=-1).eq(-1)).sum(-1) == 0).float()
            plant_truck_mask_list.append((truck_plant_resource_available.sum(-1) != 0).float())

        # shape -> (m, k)
        plant_truck_mask = torch.stack(plant_truck_mask_list, dim=0).unsqueeze(0).repeat(self.job_num, 1, 1)

        # job_task
        job_truck_mask_list = []
        job_task_duration = self.job_process_duration
        for job_id in self.job.job_capacity_state_dict.keys():
            job_resource_rep = self.job.job_capacity_state_dict[job_id]
            distance = self.job_plant_distance[job_id].unsqueeze(1).repeat(1, self.truck_num) + self.plant_truck_distance
            task_start_time = ((self.current_time - self.start_time).seconds / 60 + (distance / self.truck_travel_speed) * 60 + plant_task_duration).long()
            task_end_time = (task_start_time + job_task_duration).long()
            task_time = torch.stack([task_start_time, task_end_time], dim=2)
            task_time_job_resource_overlap = (task_time.flip(dims=[2]).view(self.plant_num, self.truck_num, 1, 1, 2).float() - job_resource_rep.unsqueeze(0).unsqueeze(0)).sign()
            truck_job_resource_available = ((task_time_job_resource_overlap.prod(dim=-1).eq(-1)).sum(-1) == 0).float()
            job_truck_mask_list.append((truck_job_resource_available.sum(-1) != 0).float())

        # shape -> (n, m, k)
        job_truck_mask = (torch.stack(job_truck_mask_list, dim=0) * (self.job.job_order_qty_tensor != 0).float().view(-1, 1, 1))

        return base_mask * job_truck_mask * plant_truck_mask

    def reset(self):
        self.current_time = datetime.strptime(self.env_config["start_time_str"], "%H:%M:%S") + timedelta(minutes=1)
        self.job_plant_distance = self._get_distance(self.coor_tensor_job, self.coor_tensor_plant).float().to(self.device)
        self.normalized_distance_job_to_plant = self._normalize(self.job_plant_distance)
        self.plant_truck_distance = torch.zeros(self.plant_num, self.truck_num).float().to(self.device)
        self.job_truck_distance = torch.zeros(self.job_num, self.truck_num).float().to(self.device)
        self.number_of_trucks_in_garage = []
        self.rewards_list = []
        self.crew.reset()
        self.job.reset()
        self.plant.reset()

        temp_obs = self._get_temp_obs()
        self.obs_hist = [temp_obs for i in range(self.env_config['hist_length'])]

        observation = self._get_observations()
        self.observation = observation

        self.execute_count = 0
        self.skip_count = 0

        return observation

    def step(self, action):
        """
        should sample action when the both resources, plants, jobs are available. if either one is not available, should let time flow forward
        until there is new action can be sampled.
        :param action:
        :return:
        """

        reward, done = self._act(action)

        temp_obs = self._get_temp_obs()
        self.obs_hist.pop(0)
        self.obs_hist.append(temp_obs)

        observation = self._get_observations()
        self.observation = observation

        return observation, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

class ConcreteProductionDeliveryEnvV2(gym.Env):
    def __init__(self, env_config_path, device='cpu'):
        assert env_config_path is not None, 'env_config_path have to be provided'
        self.env_config = yaml.load(open(env_config_path, "r"))
        self.device = device
        self.plant = Plant(
            self.env_config["plant_dict"],
            device
        )
        self.crew = Crew(
            self.env_config["crew_preserved_capacity"],
            self.env_config["max_cap_per_truck"],
            self.env_config["start_time_str"],
            device
        )
        self.job = Job(
            self.env_config["job_dict"],
            self.env_config["max_cap_per_truck"],
            device
        )
        self.start_time = datetime.strptime(self.env_config["start_time_str"], "%H:%M:%S")
        self.time_step_size = self.env_config["time_step_size"]
        self.plant_process_duration = self.env_config["plant_process_time"]
        self.job_process_duration = self.env_config["job_process_time"]
        self.truck_travel_speed = self.env_config["truck_travel_speed"]
        self.job_num = self.job.job_capacity_state_dict.__len__()
        self.plant_num = self.plant.plant_capacity_state_dict.__len__()
        self.truck_num = len(self.crew.crew_id)
        self.benchmark_truck_in_garage = [self.env_config["crew_preserved_capacity"] for i in range(10)]
        self.job_id_tensor = torch.tensor([[i, i] for i in range(self.env_config["job_dict"].__len__())]).to(self.device)
        self.plant_id_tensor = torch.tensor([[i, i] for i in range(self.env_config["job_dict"].__len__(), self.env_config['plant_dict'].__len__())]).to(self.device)
        job_qty_distribution_dict, node_node_route_distance, _, __ = data_injection_handler()
        self.node_node_route_distance = torch.from_numpy(node_node_route_distance)
        self.job_qty_distribution_dict_train, self.job_qty_distribution_dict_eval = self._job_qty_distribution_dict_split(job_qty_distribution_dict)
        self.job_job_route_distance = self.node_node_route_distance[:-self.plant_num, :-self.plant_num].to(self.device)
        self.job_plant_route_distance = self.node_node_route_distance[:-self.plant_num, -self.plant_num:].to(self.device)
        self.reset_job_order_distribution()
        self.action_space = spaces.Discrete(self.job_num * self.plant_num * self.truck_num)

    def _job_qty_distribution_dict_split(self, job_qty_distribution_dict):
        train_sample_indices = np.random.choice(np.arange(job_qty_distribution_dict.__len__()), int(0.7*job_qty_distribution_dict.__len__())).tolist()
        train_sample = {}
        eval_sample = {}
        for index, key in enumerate(job_qty_distribution_dict.keys()):
            if index in train_sample_indices:
                train_sample[key] = job_qty_distribution_dict[key]
            else:
                eval_sample[key] = job_qty_distribution_dict[key]

        return train_sample, eval_sample

    def _normalize(self, distance):
        """
        The distance here are calculated straightly by using shortest distance in 2d Euclidean space. In general, it can
        be provided with a route distance map form place to place.
        :return: tensor in shape (n, m)
        """
        min_distance_on_b = distance.min(dim=1, keepdims=True)[0]
        max_min_diff_distance_on_b = distance.max(dim=1, keepdims=True)[0] - distance.min(dim=1, keepdims=True)[0]
        normalized_distance = (distance - min_distance_on_b) / (max_min_diff_distance_on_b + 10e-5)

        return normalized_distance

    def _check_if_available(self):
        """
        True if there is available allocation of resources under constraints.
        :return:
        """
        base_mask = self.crew.crew_state_tensor.view(1, 1, -1).repeat(self.job_num, self.plant_num, 1)

        # plant task
        plant_truck_mask_list = []
        plant_task_duration = self.plant_process_duration
        for plant_id, plant_name in enumerate(self.plant.plant_capacity_state_dict.keys()):
            plant_resource_rep = self.plant.plant_capacity_state_dict[plant_id]
            task_start_time = ((self.current_time - self.start_time).seconds / 60 + (self.plant_truck_distance[plant_id] / self.truck_travel_speed) * 60).long()
            task_end_time = (task_start_time + plant_task_duration).long()
            task_time = torch.stack([task_start_time, task_end_time], dim=1)
            task_time_plant_resource_overlap = (task_time.flip(dims=[1]).view(self.truck_num, 1, 1, 2).float()  - plant_resource_rep.unsqueeze(0)).sign()
            truck_plant_resource_available = ((task_time_plant_resource_overlap.prod(dim=-1).eq(-1)).sum(-1) == 0).float()
            plant_truck_mask_list.append((truck_plant_resource_available.sum(-1) != 0).float())

        # shape -> (m, k)
        plant_truck_mask = torch.stack(plant_truck_mask_list, dim=0).unsqueeze(0).repeat(self.job_num, 1, 1)

        # job_task
        job_truck_mask_list = []
        job_task_duration = self.job_process_duration
        for job_id, job_name in enumerate(self.job.job_capacity_state_dict.keys()):
            job_resource_rep = self.job.job_capacity_state_dict[job_id]
            distance = self.job_plant_distance[job_id].unsqueeze(1).repeat(1, self.truck_num) + self.plant_truck_distance
            task_start_time = ((self.current_time - self.start_time).seconds / 60 + (distance / self.truck_travel_speed) * 60 + plant_task_duration).long()
            task_end_time = (task_start_time + job_task_duration).long()
            task_time = torch.stack([task_start_time, task_end_time], dim=2)
            task_time_job_resource_overlap = (task_time.flip(dims=[2]).view(self.plant_num, self.truck_num, 1, 1, 2).float() - job_resource_rep.unsqueeze(0).unsqueeze(0)).sign()
            truck_job_resource_available = ((task_time_job_resource_overlap.prod(dim=-1).eq(-1)).sum(-1) == 0).float()
            job_truck_mask_list.append((truck_job_resource_available.sum(-1) != 0).float())

        # shape -> (n, m, k)

        job_truck_mask = (torch.stack(job_truck_mask_list, dim=0) * (self.job.job_order_qty_tensor > 0).float().view(-1, 1, 1))

        return (((base_mask * job_truck_mask * plant_truck_mask).view(-1).sum()).item() != 0)

    def _iterate(self):
        """
        when there is no more actions available, will iterate along time flow and resource and production capacity will release along with time.
        :return:
        """
        self.crew.update(self.time_step_size)
        self.current_time += timedelta(minutes=self.time_step_size)

    def _get_temp_obs(self):
        """
        if there is no more action available after the action is taken, will iterate through time flow until there is action availble and
        return the obs when there is action available again.
        :param obs:
        :param action:
        :param reward:
        :return:
        """
        normalized_distance_job_to_truck = self._normalize(self.job_truck_distance)
        normalized_distance_plant_to_truck = self._normalize(self.plant_truck_distance)
        available_job_plant_truck = self.get_available_actions()

        job_rep_list = []
        for job_id, job_name in enumerate(self.job.job_dict.keys()):
            job_resource_rep = self.job.job_capacity_state_dict[job_id]
            job_resource_capacity_remain = (1 - (job_resource_rep[:, :, 1] - job_resource_rep[:, :, 0]).sum(-1) / (600)).mean().unsqueeze(0)
            job_order_qty_remain = self.job.job_order_qty_tensor[job_id]
            job_order_remain = (job_order_qty_remain / (self.job.job_dict[job_name][2] + 0.01)).to(self.device)
            normalized_distance_to_plant = self.normalized_distance_job_to_plant[job_id, :]
            normalized_distance_to_truck = normalized_distance_job_to_truck[job_id, :]
            process_duration = torch.tensor([self.job_process_duration / 60]).to(self.device)
            job_rep = torch.cat(
                [
                    job_resource_capacity_remain,
                    job_order_remain.unsqueeze(0),
                    normalized_distance_to_plant,
                    normalized_distance_to_truck,
                    process_duration
                ],
                dim=0
            )
            job_rep_list.append(job_rep)
        job_reps = torch.stack(job_rep_list, dim=0)

        plant_rep_list = []
        for plant_id, plant_name in enumerate(self.plant.plant_dict.keys()):
            plant_resource_rep = self.plant.plant_capacity_state_dict[plant_id]
            plant_resource_capacity_remain = (1 - (plant_resource_rep[:, :, 1] - plant_resource_rep[:, :, 0]).sum(-1) / (600)).mean().unsqueeze(0)
            normalized_distance_to_job = self.normalized_distance_job_to_plant[:, plant_id]
            normalized_distance_to_truck = normalized_distance_plant_to_truck[plant_id, :]
            process_duration = torch.tensor([self.plant_process_duration / 60]).to(self.device)
            plant_rep = torch.cat(
                [
                    plant_resource_capacity_remain,
                    normalized_distance_to_job,
                    normalized_distance_to_truck,
                    process_duration
                ],
                dim=0
            )
            plant_rep_list.append(plant_rep)
        plant_reps = torch.stack(plant_rep_list, dim=0)

        crew_distance_features = torch.cat([normalized_distance_job_to_truck.t(), normalized_distance_plant_to_truck.t()], dim=1)
        crew_rep_list = []
        # current_work_duration, standard_work_duration = self._get_work_duration()
        # work_duration = current_work_duration / standard_work_duration
        for truck_id in self.crew.crew_id:
            travel_speed = torch.tensor([self.truck_travel_speed]).float().to(self.device) / 100
            truck_rep = torch.cat(
                [
                    available_job_plant_truck[:, :, truck_id].reshape(-1),
                    crew_distance_features[truck_id, :],
                    travel_speed,
                    # torch.tensor([work_duration], dtype=torch.float).to(self.device)
                ],
                dim=0
            )
            crew_rep_list.append(
                truck_rep
            )
        crew_reps = torch.stack(crew_rep_list, dim=0)
        available_action = available_job_plant_truck.view(-1)
        current_work_duration, standard_work_duration = self._get_work_duration()
        work_duration = current_work_duration / standard_work_duration
        order_remain = (self.job.job_order_qty_tensor.sum() / self.job.job_initial_order_qty).to(self.device)
        miscellaneous_rep = torch.cat(
            [
                available_action,
                torch.tensor([work_duration], dtype=torch.float).to(self.device),
                order_remain.unsqueeze(0),
                self.crew.crew_state_tensor.to(self.device)
            ],
            dim=0
        )
        # miscellaneous_rep = torch.tensor([0]).to(self.device)

        temp_obs = tuple([job_reps.numpy(), plant_reps.numpy(), crew_reps.numpy(), miscellaneous_rep.numpy()])

        return temp_obs

    # def _get_observations(self):
    #     job_reps_hist = []
    #     plant_reps_hist = []
    #     crew_reps_hist = []
    #     misc_reps_hist = []
    #     for obs in self.obs_hist:
    #         job_reps_hist.append(obs[0])
    #         plant_reps_hist.append(obs[1])
    #         crew_reps_hist.append(obs[2])
    #         misc_reps_hist.append(obs[3])
    #
    #     observation = tuple(
    #         [
    #             torch.stack(job_reps_hist, dim=0),
    #             torch.stack(plant_reps_hist, dim=0),
    #             torch.stack(crew_reps_hist, dim=0),
    #             torch.stack(misc_reps_hist, dim=0),
    #         ])
    #
    #     return observation

    def _get_work_duration(self):
        current_work_duration = (self.current_time - datetime.strptime(self.env_config["start_time_str"], "%H:%M:%S")).seconds / 60
        standard_work_duration = (datetime.strptime(self.env_config["off_time_str"], "%H:%M:%S") - datetime.strptime(self.env_config["start_time_str"], "%H:%M:%S")).seconds / 60

        return current_work_duration, standard_work_duration

    def _act(self, action):
        """
        task duration need to compute by using the coor id and production time. action space will only include the available actions can be made.
        The distance here are calculated straightly by using shortest distance in 2d Euclidean space. In general, it can
        be provided with a route distance map form place to place.
        :param observation:
        :param action: int -> action id
        :return: reward
        """
        if action[0] == 0:
            selected_job, selected_plant, selected_truck = action[1:]
            if self.get_available_actions()[selected_job, selected_plant, selected_truck].item() == 0:
                self._iterate()
                reward = -1
            else:
                if int(self.crew.crew_coor_tensor[selected_truck, 0].item()) == -1:
                    plant_truck_dist = 0
                else:
                    plant_truck_dist = self.job_plant_route_distance[int(self.crew.crew_coor_tensor[selected_truck, 0].item()), selected_plant]
                plant_job_dist = self.job_plant_route_distance[selected_job, selected_plant]

                # execute truck
                truck_task_duration = int(((plant_job_dist + plant_truck_dist) / self.truck_travel_speed) * 60 + self.plant_process_duration  + self.job_process_duration)
                self.crew.execute(selected_truck, truck_task_duration, self.job_id_tensor[selected_job])

                # execute plant
                plant_task_duration = self.plant_process_duration
                plant_task_start_time = int((self.current_time - self.start_time).seconds / 60 + (plant_truck_dist / self.truck_travel_speed) * 60)
                self.plant.execute(selected_plant, plant_task_duration, plant_task_start_time)

                # execute job
                job_task_duration = self.job_process_duration
                job_task_start_time = int((self.current_time - self.start_time).seconds / 60 + ((plant_job_dist + plant_truck_dist) / self.truck_travel_speed) * 60 + self.plant_process_duration)
                self.job.execute(selected_job, job_task_duration, job_task_start_time)

                self.job_truck_distance[:, selected_truck] = self.job_job_route_distance[:, int(self.crew.crew_coor_tensor[selected_truck, 0].item())]
                self.plant_truck_distance[:, selected_truck] = self.job_plant_route_distance[int(self.crew.crew_coor_tensor[selected_truck, 0].item()), :]

                truck_cost_bonus = (self.crew.number_of_trucks_in_garage().item() / self.env_config['crew_preserved_capacity'])
                remain_qty_penalty = self.job.job_order_qty_tensor.sum().item()/self.job.job_initial_order_qty.sum().item()
                time_factor = math.exp(((self.current_time - self.start_time).seconds / (datetime.strptime(self.env_config['max_time_str'], "%H:%M:%S") - self.start_time).seconds) - 1)

                reward = (truck_cost_bonus - remain_qty_penalty) * time_factor

        elif action[0] == 1:
            self._iterate()
            reward = 0

        else:
            raise ValueError('action[0] should be either 0 or 1')

        self.number_of_trucks_in_garage.append(self.crew.number_of_trucks_in_garage().item())
        self.execute_count += 1

        if self.job.job_order_qty_tensor.sum() == 0:
            done = True
        else:
            done = False

        if not done:
            if self.current_time + timedelta(minutes=self.crew.crew_task_duration_tensor.max(dim=0)[0].item()) >= datetime.strptime(self.env_config['max_time_str'], "%H:%M:%S"):
                done = True

            while not self._check_if_available():
                self._iterate()
                if self.current_time + timedelta(minutes=self.crew.crew_task_duration_tensor.max(dim=0)[0].item()) >= datetime.strptime(self.env_config['max_time_str'], "%H:%M:%S"):
                    done = True

        if done:
            if self.current_time + timedelta(minutes=self.crew.crew_task_duration_tensor.max(dim=0)[0].item()) < datetime.strptime(self.env_config['max_time_str'], "%H:%M:%S"):
                reward += 0

            else:
                reward -= max([1, 5*(self.job.job_order_qty_tensor.sum().item()/self.job.job_initial_order_qty.sum().item())])
                # reward -= 1

        return reward, done

    def reset_job_order_distribution(self, train=True):
        job_dict = self.env_config["job_dict"].copy()
        if train:
            # if random.random() < 0.5:
            order_qty = random.randint(2500, 4000)
            order_qty_distribution = (torch.rand(job_dict.keys().__len__()).softmax(dim=0) * order_qty).int().tolist()
            # else:
            #     order_qty_distribution = self.job_qty_distribution_dict_train[np.random.choice(list(self.job_qty_distribution_dict_train.keys()))].tolist()
        else:
            order_qty_distribution = self.job_qty_distribution_dict_eval[np.random.choice(list(self.job_qty_distribution_dict_eval.keys()))].tolist()

        for job_id, job_name in enumerate(job_dict.keys()):
            job_order_qty = order_qty_distribution[job_id]
            job_dict[job_name][2] = job_order_qty

        self.job = Job(
            job_dict,
            self.env_config["max_cap_per_truck"],
            self.device
        )
        self.truck_travel_speed = random.randint(40, 80)
        observation = self.reset()

        return observation

    def get_eval_info(self):
        current_work_duration, standard_work_duration = self._get_work_duration()
        ot_duration = current_work_duration + self.crew.crew_task_duration_tensor.max(dim=0)[0].item() - standard_work_duration

        max_number_of_trucks_on_road = self.env_config["crew_preserved_capacity"] - min(self.number_of_trucks_in_garage)

        qty_remain_ratio = (self.job.job_order_qty_tensor.sum() / self.job.job_initial_order_qty.sum()).item()

        number_of_trucks_on_road = self.env_config['crew_preserved_capacity'] -  min(self.number_of_trucks_in_garage) + 0.0001
        order_qty_delivered = (self.job.job_initial_order_qty.sum() - self.job.job_order_qty_tensor.sum()).item()
        truck_on_road_productivity = (order_qty_delivered / self.env_config['max_cap_per_truck']) / number_of_trucks_on_road

        return max_number_of_trucks_on_road, ot_duration, qty_remain_ratio, truck_on_road_productivity

    def get_obs_info(self):
        job_dim = self.observation[0].shape[1]
        plant_dim = self.observation[1].shape[1]
        crew_dim = self.observation[2].shape[1]
        misc_info_dim = self.observation[3].shape[0]
        # hist_length = self.observation[0].shape[0]

        return job_dim, plant_dim, crew_dim, misc_info_dim, self.job_num, self.plant_num, self.truck_num, None

    def get_task_duration(self, selected_job, selected_plant, selected_truck):
        if int(self.crew.crew_coor_tensor[selected_truck, 0].item()) == -1:
            plant_truck_dist = 0
        else:
            plant_truck_dist = self.job_plant_route_distance[int(self.crew.crew_coor_tensor[selected_truck, 0].item()), selected_plant]
        plant_job_dist = self.job_plant_route_distance[selected_job, selected_plant]
        task_duration = ((plant_job_dist + plant_truck_dist) / self.truck_travel_speed)*60 + self.plant_process_duration  + self.job_process_duration

        return task_duration

    def get_available_actions(self):
        base_mask = self.crew.crew_state_tensor.view(1, 1, -1).repeat(self.job_num, self.plant_num, 1)
        # print(self.crew.crew_state_tensor)

        # plant task
        plant_truck_mask_list = []
        plant_task_duration = self.plant_process_duration
        for plant_id, plant_name in enumerate(self.plant.plant_capacity_state_dict.keys()):
            plant_resource_rep = self.plant.plant_capacity_state_dict[plant_id]
            task_start_time = ((self.current_time - self.start_time).seconds / 60 + (self.plant_truck_distance[plant_id] / self.truck_travel_speed) * 60).long()
            task_end_time = (task_start_time + plant_task_duration).long()
            task_time = torch.stack([task_start_time, task_end_time], dim=1)
            task_time_plant_resource_overlap = (task_time.flip(dims=[1]).view(self.truck_num, 1, 1, 2).float()  - plant_resource_rep.unsqueeze(0)).sign()
            truck_plant_resource_available = ((task_time_plant_resource_overlap.prod(dim=-1).eq(-1)).sum(-1) == 0).float()
            plant_truck_mask_list.append((truck_plant_resource_available.sum(-1) != 0).float())

        # shape -> (m, k)
        plant_truck_mask = torch.stack(plant_truck_mask_list, dim=0).unsqueeze(0).repeat(self.job_num, 1, 1)

        # job_task
        job_truck_mask_list = []
        job_task_duration = self.job_process_duration
        for job_id, job_name in enumerate(self.job.job_capacity_state_dict.keys()):
            job_resource_rep = self.job.job_capacity_state_dict[job_id]
            distance = self.job_plant_distance[job_id].unsqueeze(1).repeat(1, self.truck_num) + self.plant_truck_distance
            task_start_time = ((self.current_time - self.start_time).seconds / 60 + (distance / self.truck_travel_speed) * 60 + plant_task_duration).long()
            task_end_time = (task_start_time + job_task_duration).long()
            task_time = torch.stack([task_start_time, task_end_time], dim=2)
            task_time_job_resource_overlap = (task_time.flip(dims=[2]).view(self.plant_num, self.truck_num, 1, 1, 2).float() - job_resource_rep.unsqueeze(0).unsqueeze(0)).sign()
            truck_job_resource_available = ((task_time_job_resource_overlap.prod(dim=-1).eq(-1)).sum(-1) == 0).float()
            job_truck_mask_list.append((truck_job_resource_available.sum(-1) != 0).float())

        # shape -> (n, m, k)
        job_truck_mask = (torch.stack(job_truck_mask_list, dim=0) * (self.job.job_order_qty_tensor != 0).float().view(-1, 1, 1))

        return base_mask * job_truck_mask * plant_truck_mask

    def reset(self):
        self.current_time = datetime.strptime(self.env_config["start_time_str"], "%H:%M:%S") + timedelta(minutes=1)
        self.job_plant_distance = self.job_plant_route_distance.float().to(self.device)
        self.normalized_distance_job_to_plant = self._normalize(self.job_plant_distance)
        self.plant_truck_distance = torch.zeros(self.plant_num, self.truck_num).float().to(self.device)
        self.job_truck_distance = torch.zeros(self.job_num, self.truck_num).float().to(self.device)
        self.number_of_trucks_in_garage = []
        self.rewards_list = []
        self.crew.reset()
        self.job.reset()
        self.plant.reset()

        observation = self._get_temp_obs()
        # self.obs_hist = [temp_obs for i in range(self.env_config['hist_length'])]
        #
        # observation = self._get_observations()
        self.observation = observation

        self.execute_count = 0
        self.skip_count = 0

        return observation

    def step(self, action):
        """
        should sample action when the both resources, plants, jobs are available. if either one is not available, should let time flow forward
        until there is new action can be sampled.
        :param action:
        :return:
        """

        reward, done = self._act(action)

        observation = self._get_temp_obs()
        # self.obs_hist.pop(0)
        # self.obs_hist.append(temp_obs)

        # observation = self._get_observations()
        self.observation = observation

        return observation, reward, done, None

    def render(self, mode='human'):
        pass

    def close(self):
        pass
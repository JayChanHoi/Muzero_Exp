from .concrete_production_delivery_env import ConcreteProductionDeliveryEnv, ConcreteProductionDeliveryEnvV2
from itertools import count
import os

import torch
import random

if __name__ == '__main__':
    # Env = ConcreteProductionDeliveryEnv('/Users/jaychan/PycharmProjects/Gammon/concrete_factory/concrete-scheduling/src/config/env_config.yml', device='cpu')
    Env = ConcreteProductionDeliveryEnvV2(os.path.join(os.path.dirname(__file__),'env_config_real_data.yml'), device='cpu')
    state = Env.reset_job_order_distribution()
    job_feature_dim, plant_feature_dim, crew_feature_dim, misc_info_feature_dim, job_num, plant_num, truck_num, hist_length = Env.get_obs_info()
    # policy_network = PolicyOptimizationNet(
    #     0.1,
    #     job_feature_dim,
    #     plant_feature_dim,
    #     crew_feature_dim,
    #     misc_info_feature_dim,
    #     plant_num,
    #     truck_num,
    #     job_num=job_num,
    #     value_net=False
    # )

    # agent_core_net = QNet(
    #     0.1,
    #     job_feature_dim,
    #     plant_feature_dim,
    #     crew_feature_dim,
    #     misc_info_feature_dim,
    #     plant_num,
    #     truck_num,
    #     job_num=job_num,
    #     hist_length=hist_length
    # )


    iter_count = 0

    with torch.no_grad():
        for iter in count():

            # if random.random() < 0.5:
            action = torch.cat([torch.tensor([1], dtype=torch.float), Env.get_available_actions().float().view(-1)], dim=0).multinomial(1).view(1)
            # print(available_job_plant_truck.sum())
            # print(available_job_plant_truck.nonzero())
            # else:
            #     action_value, _ = agent_core_net([[job_feature, plant_feature, crew_feature, misc_info_feature], [available_job, available_plant, available_truck], None])
            #     action = action_value.argmax(dim=1)
            #     print(job_feature)
            #     print(action_value.shape)
            #     print(available_job_plant_truck.shape)
                # print(action_value[0, action])
            # action = torch.cat([torch.tensor([1], dtype=torch.float), available_job_plant_truck.float().view(-1)], dim=0).multinomial(1).view(1)
            # print(action)

            # if_iterate = action.item() // (plant_num * truck_num * job_num)
            # selected_job = (action.item() - if_iterate * (plant_num * truck_num * job_num)) // (plant_num * truck_num)
            # selected_plant = (action.item() - if_iterate * (plant_num * truck_num * job_num) - selected_job * (plant_num * truck_num)) // truck_num
            # selected_crew = action.item() - if_iterate * (plant_num * truck_num * job_num) - selected_job * (plant_num * truck_num) - selected_plant * truck_num

            if action.item() == 0:
                if_iterate = 1
                selected_job = 0
                selected_plant = 0
                selected_crew = 0
            else:
                if_iterate = 0
                selected_job = (action.item() - 1) // (plant_num * truck_num)
                selected_plant = (action.item() - 1 - selected_job * (plant_num * truck_num)) // truck_num
                selected_crew = action.item() - 1 - selected_job * (plant_num * truck_num) - selected_plant * truck_num
                
            state, reward, done, _ = Env.step((if_iterate, selected_job, selected_plant, selected_crew))
            iter_count += 1
            print('action', action)
            print('time', Env.current_time)
            print('number of  delivery ', (Env.job.job_initial_order_qty.sum() - Env.job.job_order_qty_tensor.sum()).item() / Env.env_config['max_cap_per_truck'])
            print('number of truck on road', Env.env_config['crew_preserved_capacity'] - Env.crew.number_of_trucks_in_garage().item())
            print('job_order_qty', Env.job.job_order_qty_tensor.sum().item())
            print('reward', reward)
            print('------------------------------------------------------------------------------------------------')
            if done:
                print(Env.get_eval_info())
                print('episode length: {}'.format(iter_count))
                break
#
#
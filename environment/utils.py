import pandas as pd
import numpy as np

import os
import yaml
from tqdm import tqdm

def extract_info():
    job_path = os.path.join(os.path.dirname(__file__),'data/wo_delivery_extra_project.xlsx')
    df = pd.read_excel(job_path)
    df_refine = df[['JOB_NO', 'DELIVERY_DATE', 'ORDER_QTY']]

    new_df_data = {'DELIVERY_DATE':[]}
    for job_no in set(df_refine['JOB_NO']):
        new_df_data[job_no] = []

    for date in set(df_refine['DELIVERY_DATE']):
        new_df_data['DELIVERY_DATE'].append(date)
        sub_df = df_refine[df_refine['DELIVERY_DATE'] == date]
        for key in new_df_data.keys():
            if key != 'DELIVERY_DATE':
                if key not in set(sub_df['JOB_NO']):
                    new_df_data[key].append(0)
                else:
                    new_df_data[key].append(sub_df[sub_df['JOB_NO'] == key]['ORDER_QTY'].values[0])

    new_df = pd.DataFrame(data=new_df_data)
    new_df.to_excel(os.path.join(os.path.dirname(__file__),'data/job_qty_distribution.xlsx'), index=False)

    job_dict = {col:index for index, col in enumerate(new_df.columns) if col != "DELIVERY_DATE"}

    job_name_file_path = os.path.join(os.path.dirname(__file__),'data/job_no_&_job_name.xlsx')
    job_name_df = pd.read_excel(job_name_file_path)
    refine_job_name_dict = {'job_id':[], 'job_name':[]}
    missing_job_id = []
    for job_id in job_dict.keys():
        if job_id not in job_name_df['job_id'].tolist():
            missing_job_id.append(job_id)
        else:
            refine_job_name_dict['job_id'].append(job_id)
            refine_job_name_dict['job_name'] += job_name_df[job_name_df['job_id'] == job_id]['job_name'].values.tolist()
    refine_job_name_df = pd.DataFrame(data=refine_job_name_dict)
    refine_job_name_df.to_excel(os.path.join(os.path.dirname(__file__),'data/target_job_name.xlsx'), index=False)

def get_route_distance(API_KEY):
    import googlemaps

    job_plant_address_file_path = os.path.join(os.path.dirname(__file__),'data/plant_job_name.xlsx')
    job_plant_address_df = pd.read_excel(job_plant_address_file_path)[['ID','Address']]
    node_to_node_route_distance = np.zeros((len(job_plant_address_df), len(job_plant_address_df)))
    node_to_node_route_duration = np.zeros((len(job_plant_address_df), len(job_plant_address_df)))
    gmaps = googlemaps.Client(key=API_KEY)
    memory_dict = {}
    for i in tqdm(range(len(job_plant_address_df))):
        for j in range(len(job_plant_address_df)):
            if i == j:
                node_to_node_route_distance[i, j] = 0
                node_to_node_route_duration[i, j] = 0
            elif job_plant_address_df.iloc[i]['Address'] == job_plant_address_df.iloc[j]['Address']:
                node_to_node_route_distance[i, j] = 0
                node_to_node_route_duration[i, j] = 0
            elif tuple([j,i]) in memory_dict.keys():
                node_to_node_route_distance[i, j] = memory_dict[tuple([j, i])][0]
                node_to_node_route_duration[i, j] = memory_dict[tuple([j, i])][1]
            else:
                response = gmaps.distance_matrix(job_plant_address_df.iloc[i]['Address'], job_plant_address_df.iloc[j]['Address'])
                print(response)
                distance = response['rows'][0]['elements'][0]['distance']['text'].replace(' ', '').replace('m', '').replace('k', '')
                duration = response['rows'][0]['elements'][0]['duration']['text'].replace(' ', '').replace('min', '').replace('s', '')
                node_to_node_route_distance[i, j] = distance
                node_to_node_route_duration[i, j] = duration
                memory_dict[tuple([i, j])] = [distance, duration]

    node_to_node_route_distance_df = pd.DataFrame(node_to_node_route_distance, columns=job_plant_address_df['ID'])
    node_to_node_route_duration_df = pd.DataFrame(node_to_node_route_duration, columns=job_plant_address_df['ID'])

    node_to_node_route_distance_df.to_excel(os.path.join(os.path.dirname(__file__),'data/node_to_node_route_distance.xlsx'), index=False)
    node_to_node_route_duration_df.to_excel(os.path.join(os.path.dirname(__file__),'data/node_to_node_route_duration.xlsx'), index=False)


def data_injection_handler():
    node_node_route_distance_file_path = os.path.join(os.path.dirname(__file__),'data/node_to_node_route_distance_v2.xlsx')
    node_node_route_distance_df = pd.read_excel(node_node_route_distance_file_path)

    job_qty_distribution_file_path = os.path.join(os.path.dirname(__file__),'data/trimmed_job_qty_distribution_v2.xlsx')
    job_qty_distribution_df = pd.read_excel(job_qty_distribution_file_path)

    job_qty_distribution_dict = {}
    for index, row in job_qty_distribution_df.iterrows():
        row_dict = row.to_dict()
        date = row_dict['DELIVERY_DATE'].date()
        qty_distribution = np.array(list(row_dict.values())[1:])
        if qty_distribution.sum() > 1000:
            job_qty_distribution_dict[date] = qty_distribution

    job_list = job_qty_distribution_df.columns[1:].tolist()
    plant_list = node_node_route_distance_df.columns[-3:].tolist()

    node_node_route_distance = np.array(node_node_route_distance_df)[:, 1:].astype(np.float32)

    return job_qty_distribution_dict, node_node_route_distance, job_list, plant_list

def generate_env_config_yml():
    """
    crew_preserved_capacity: 150
    max_cap_per_truck: 7
    start_time_str: "08:30:00"
    off_time_str: "17:30:00"
    max_time_str: "19:00:00"
    time_step_size: 2
    truck_travel_speed: 1000
    process_time: 20
    work_duration_upperbound: 960
    hist_length: 4
    :return:
    """

    _ , __, job_list, plant_list = data_injection_handler()
    env_config ={
        'job_dict':{job:[0, 6, 0] for job in job_list},
        'plant_dict':{plant:[0, 15] for plant in plant_list},
        'crew_preserved_capacity':150,
        'max_cap_per_truck': 7,
        'start_time_str': "08:30:00",
        'off_time_str': "17:30:00",
        'max_time_str': "19:00:00",
        'time_step_size': 2,
        'truck_travel_speed': 50,
        'process_time': 20,
        'work_duration_upperbound': 960,
        'hist_length': 4
    }

    env_config_main_root = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    with open(os.path.join(env_config_main_root, 'config/env_config_real_data.yml'), 'w') as file:
        yaml.dump(env_config, file)


if __name__ == '__main__':
    # pass
    # extract_info()
    # data_injection_handler()
    generate_env_config_yml()
    # get_route_distance()
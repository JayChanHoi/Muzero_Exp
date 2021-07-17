from datetime import datetime, timedelta, time
import torch

class AbstractAuxiClass(object):
    def __init__(self, *args, **kargs):
        pass

    def reset(self):
        pass

    def _release(self, *args):
        pass

    def execute(self, *args):
        pass

    def check_if_conflict(self, *args):
        pass

    def update(self, *args):
        pass

    def find_capacity(self, *args):
        pass

class Crew(AbstractAuxiClass):
    def __init__(self, preserved_capacity=100, max_cap_per_truck=7, start_time_str="08:30:00", device='cpu'):
        super(Crew, self).__init__()
        self.start_time_str = start_time_str
        self.preserved_capacity = preserved_capacity
        self.max_cap_per_truck = max_cap_per_truck
        self.device = device
        self.reset()

    def reset(self):
        self.crew_id = tuple([truck_id for truck_id in range(self.preserved_capacity)])
        self.current_time = datetime.strptime(self.start_time_str, "%H:%M:%S") + timedelta(minutes=1)
        self.crew_coor_tensor = -1 * torch.ones(len(self.crew_id), 2).float().to(self.device)
        self.crew_state_tensor = torch.ones(len(self.crew_id)).float().to(self.device)
        self.crew_task_duration_tensor = torch.zeros(len(self.crew_id)).float().to(self.device)

    def number_of_trucks_in_garage(self):
        return self.crew_coor_tensor[:, 0].eq(-1).sum()

    def find_num_of_trucks_in_use(self):
        return self.crew_state_tensor.eq(0).sum()

    def find_num_of_trucks_available(self):
        return self.crew_state_tensor.eq(1).sum()

    def execute(self, truck_id, task_duration, coor_id):
        if not truck_id in self.crew_state_tensor.eq(1).nonzero().view(-1).tolist():
            raise ValueError('the truck with {} is not in the available list'.format(truck_id))

        self.crew_coor_tensor[truck_id] = coor_id
        self.crew_state_tensor[truck_id] = 0
        self.crew_task_duration_tensor[truck_id] = task_duration

    def update(self, time_step_size):
        self.crew_task_duration_tensor -= time_step_size
        self.crew_task_duration_tensor.clamp_(0)
        self.crew_state_tensor = self.crew_task_duration_tensor.eq(0).float()

        self.current_time += timedelta(minutes=time_step_size)

    def find_capacity(self, plant_coor_id):
        mask_1 = self.crew_coor_tensor.eq(-1)
        mask_2 = self.crew_coor_tensor.eq(plant_coor_id)*(self.crew_state_tensor.eq(1)+self.crew_state_tensor.eq(0)*self.crew_task_duration_tensor.le(10))
        mask = (mask_1 + mask_2).ge(1)

        return mask.sum(), torch.masked_select(torch.tensor(list(self.crew_id)), mask).float(), mask

class Plant(AbstractAuxiClass):
    def __init__(self, plant_dict, device='cpu'):
        super(Plant, self).__init__()
        self.plant_dict = plant_dict
        self.device = device
        self.resource_record_num = 200
        self.reset()

    def reset(self):
        self.plant_capacity_state_dict = self._init_plant_capacity_state_dict()

    def _init_plant_capacity_state_dict(self):
        plant_capacity_state_dict = {}
        for plant_id, (_, plant_capacity) in enumerate(self.plant_dict.values()):
            plant_capacity_state_dict[plant_id] = torch.zeros(plant_capacity, self.resource_record_num, 2).float().to(self.device)

        return plant_capacity_state_dict

    def execute(self, plant_id, task_duration, task_start_time):
        plant_resource_rep = self.plant_capacity_state_dict[plant_id]
        task_end_time = task_start_time + task_duration
        task_time_rep = torch.tensor([task_start_time, task_end_time]).float().to(self.device)
        task_time_resource_overlap = (task_time_rep.flip(dims=[0]).view(1, 1, 2)  - plant_resource_rep).sign()
        resource_available_mask = ((task_time_resource_overlap.prod(dim=-1).eq(-1)).sum(-1) == 0).float()

        if resource_available_mask.view(-1).sum().item() == 0:
            raise ValueError("plant {} has no resource for production".format(plant_id))

        resource_record_num = (plant_resource_rep.sum(dim=-1) != 0).sum(dim=-1)
        resource_id = (resource_available_mask.view(-1) * (resource_record_num < self.resource_record_num)).float().multinomial(1)
        self.plant_capacity_state_dict[plant_id][resource_id, resource_record_num[resource_id]] = task_time_rep

class Job(AbstractAuxiClass):
    def __init__(self, job_dict, max_cap_per_truck=7, device='cpu'):
        super(Job, self).__init__()
        self.job_dict = job_dict
        self.max_cap_per_truck = max_cap_per_truck
        self.device = device
        self.resource_record_num = 50
        self.reset()

    def reset(self):
        self.job_order_qty_tensor = torch.tensor([job_order_qty for _, __, job_order_qty in self.job_dict.values()]).float().to(self.device)
        self.job_capacity_state_dict = self._init_job_capacity_state_dict()
        self.job_initial_order_qty = self.job_order_qty_tensor.sum()

    def _init_job_capacity_state_dict(self):
        job_capacity_state_dict = {}
        for job_id, (_, job_capacity, __) in enumerate(self.job_dict.values()):
            job_capacity_state_dict[job_id] = torch.zeros(job_capacity, self.resource_record_num, 2).to(self.device)

        return job_capacity_state_dict

    def execute(self, job_id, task_duration, task_start_time):
        job_resource_rep = self.job_capacity_state_dict[job_id]
        task_end_time = task_start_time + task_duration
        task_time_rep = torch.tensor([task_start_time, task_end_time]).float().to(self.device)
        task_time_resource_overlap = (task_time_rep.flip(dims=[0]).view(1, 1, 2)  - job_resource_rep).sign()
        resource_available_mask = ((task_time_resource_overlap.prod(dim=-1).eq(-1)).sum(-1) == 0).float()

        if resource_available_mask.view(-1).sum().item() == 0:
            raise ValueError("job {} has no resource for production".format(job_id))

        resource_record_num = (job_resource_rep.sum(dim=-1) != 0).sum(dim=-1)
        resource_id = (resource_available_mask.view(-1) * (resource_record_num < self.resource_record_num)).float().multinomial(1)
        self.job_capacity_state_dict[job_id][resource_id, resource_record_num[resource_id]] = task_time_rep

        self.job_order_qty_tensor[job_id] -= self.max_cap_per_truck
        self.job_order_qty_tensor.clamp_(0)


"""
The MIT License

Copyright (c) 2020 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


####################################
# EXTERNAL LIBRARY
####################################
import torch
import numpy as np

# For debugging
from IPython.core.debugger import set_trace

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


####################################
# PROJECT VARIABLES
####################################
from HYPER_PARAMS import *
from TORCH_OBJECTS import *


####################################
# DATA
####################################
def CVRP_DATA_LOADER__RANDOM(num_sample, num_nodes, batch_size):
    dataset = CVRP_Dataset__Random(num_sample=num_sample, num_nodes=num_nodes)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=CVRP_collate_fn)
    return data_loader


class CVRP_Dataset__Random(Dataset):
    def __init__(self, num_sample, num_nodes):
        self.num_sample = num_sample
        self.num_nodes = num_nodes

        if num_nodes == 20:
            demand_scaler = 30
        elif num_nodes == 50:
            demand_scaler = 40
        elif num_nodes == 100:
            demand_scaler = 50
        else:
            raise NotImplementedError
        self.demand_scaler = demand_scaler

    def __getitem__(self, index):
        depot_xy_data = np.random.rand(1, 2)
        node_xy_data = np.random.rand(self.num_nodes, 2)
        node_demand_data = np.random.randint(1, 10, self.num_nodes) / self.demand_scaler

        return depot_xy_data, node_xy_data, node_demand_data

    def __len__(self):
        return self.num_sample


def CVRP_collate_fn(batch):
    depot_xy_data_tuples, node_xy_data_tuples, node_demand_data_tuples = zip(*batch)
    depot_xy = Tensor(depot_xy_data_tuples)
    node_xy = Tensor(node_xy_data_tuples)
    node_demand = Tensor(node_demand_data_tuples)[:, :, None]  # unsqeeeze to match the shape of node_xy
    return depot_xy, node_xy, node_demand


####################################
# STATE
####################################
class STATE:

    def __init__(self, data):
        # data.shape = (batch, problem+1, 3)

        self.batch_s = data.size(0)
        self.data = data

        # History
        ####################################
        self.selected_count = 0
        self.current_node = None
        # shape = (batch,)
        self.selected_node_list = LongTensor(np.zeros((self.batch_s, 0)))
        # shape = (batch, selected_count)

        # Status
        ####################################
        self.at_the_depot = None
        # shape = (batch,)
        self.loaded = Tensor(np.ones((self.batch_s,)))
        # shape = (batch,)
        self.visited_ninf_flag = Tensor(np.zeros((self.batch_s, PROBLEM_SIZE+1)))
        # shape = (batch, problem+1)
        self.ninf_mask = Tensor(np.zeros((self.batch_s, PROBLEM_SIZE+1)))
        # shape = (batch, problem+1)
        self.finished = BoolTensor(np.zeros((self.batch_s,)))
        # shape = (batch,)

    def move_to(self, selected_node_idx):
        # selected_node_idx.shape = (batch,)

        # History
        ####################################
        self.selected_count += 1
        self.current_node = selected_node_idx
        self.selected_node_list = torch.cat((self.selected_node_list, selected_node_idx[:, None]), dim=1)

        # Status
        ####################################
        self.at_the_depot = (selected_node_idx == 0)
        demand_list = self.data[:, :, 2]
        # shape = (batch, problem+1)
        gathering_index = selected_node_idx[:, None]
        selected_demand = demand_list.gather(dim=1, index=gathering_index).squeeze(dim=1)
        # shape = (batch,)
        self.loaded -= selected_demand
        self.loaded[self.at_the_depot] = 1 # refill loaded at the depot
        self.visited_ninf_flag[torch.arange(self.batch_s), selected_node_idx] = -np.inf
        self.finished = self.finished + (self.visited_ninf_flag == -np.inf).all(dim=1)
        # shape = (batch,)

        # Status Edit
        ####################################
        self.visited_ninf_flag[:, 0][~self.at_the_depot] = 0  # allow car to visit depot anytime
        demand_too_large = self.loaded[:, None] < demand_list
        # shape = (batch, problem+1)
        self.ninf_mask = self.visited_ninf_flag.clone()
        self.ninf_mask[demand_too_large] = -np.inf

        self.ninf_mask[self.finished[:, None].expand(self.batch_s, PROBLEM_SIZE+1)] = 0  # do not mask finished episode



class GROUP_STATE:

    def __init__(self, group_size, data):
        # data.shape = (batch, problem+1, 3)

        self.batch_s = data.size(0)
        self.group_s = group_size
        self.data = data

        # History
        ####################################
        self.selected_count = 0
        self.current_node = None
        # shape = (batch, group)
        self.selected_node_list = LongTensor(np.zeros((self.batch_s, self.group_s, 0)))
        # shape = (batch, group, selected_count)

        # Status
        ####################################
        self.at_the_depot = None
        # shape = (batch, group)
        self.loaded = Tensor(np.ones((self.batch_s, self.group_s)))
        # shape = (batch, group)
        self.visited_ninf_flag = Tensor(np.zeros((self.batch_s, self.group_s, PROBLEM_SIZE+1)))
        # shape = (batch, group, problem+1)
        self.ninf_mask = Tensor(np.zeros((self.batch_s, self.group_s, PROBLEM_SIZE+1)))
        # shape = (batch, group, problem+1)
        self.finished = BoolTensor(np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)

    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # History
        ####################################
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat((self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)

        # Status
        ####################################
        self.at_the_depot = (selected_idx_mat == 0)
        demand_list = self.data[:, None, :, 2].expand(self.batch_s, self.group_s, -1)
        # shape = (batch, group, problem+1)
        gathering_index = selected_idx_mat[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape = (batch, group)
        self.loaded -= selected_demand
        self.loaded[self.at_the_depot] = 1 # refill loaded at the depot
        batch_idx_mat = torch.arange(self.batch_s)[:, None].expand(self.batch_s, self.group_s)
        group_idx_mat = torch.arange(self.group_s)[None, :].expand(self.batch_s, self.group_s)
        self.visited_ninf_flag[batch_idx_mat, group_idx_mat, selected_idx_mat] = -np.inf
        self.finished = self.finished + (self.visited_ninf_flag == -np.inf).all(dim=2)
        # shape = (batch, group)

        # Status Edit
        ####################################
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # allow car to visit depot anytime
        round_error_epsilon = 0.000001
        demand_too_large = self.loaded[:, :, None] + round_error_epsilon < demand_list
        # shape = (batch, group, problem+1)
        self.ninf_mask = self.visited_ninf_flag.clone()
        self.ninf_mask[demand_too_large] = -np.inf

        self.ninf_mask[self.finished[:, :, None].expand(self.batch_s, self.group_s, PROBLEM_SIZE+1)] = 0
        # do not mask finished episode


####################################
# ENVIRONMENT
####################################
class ENVIRONMENT:

    def __init__(self, depot_xy, node_xy, node_demand):
        # depot_xy.shape = (batch, 1, 2)
        # node_xy.shape = (batch, problem, 2)
        # node_demand.shape = (batch, problem, 1)

        self.batch_s = depot_xy.size(0)
        self.state = None

        all_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape = (batch, problem+1, 2)
        depot_demand = Tensor(np.zeros((self.batch_s, 1, 1)))
        all_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape = (batch, problem+1, 1)
        self.data = torch.cat((all_node_xy, all_node_demand), dim=2)
        # shape = (batch, problem+1, 3)

    def reset(self):
        self.state = STATE(self.data)

        reward = None
        done = False
        return self.state, reward, done

    def step(self, selected_node_idx):
        # selected_node_idx.shape = (batch,)

        # move state
        self.state.move_to(selected_node_idx)

        # returning values
        done = self.state.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.state, reward, done

    def _get_travel_distance(self):
        all_node_xy = self.data[:, :, 0:2]
        # shape = (batch, problem+1, 2)
        gathering_index = self.state.selected_node_list.unsqueeze(2).expand(-1, -1, 2)
        # shape = (batch, selected_count, 2)
        ordered_seq = all_node_xy.gather(dim=1, index=gathering_index)
        # shape = (batch, selected_count, 2)

        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(2).sqrt()
        # size = (batch, selected_count)

        travel_distances = segment_lengths.sum(1)
        # size = (batch,)
        return travel_distances


class GROUP_ENVIRONMENT:

    def __init__(self, depot_xy, node_xy, node_demand):
        # depot_xy.shape = (batch, 1, 2)
        # node_xy.shape = (batch, problem, 2)
        # node_demand.shape = (batch, problem, 1)

        self.batch_s = depot_xy.size(0)
        self.group_s = None
        self.group_state = None

        all_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape = (batch, problem+1, 2)
        depot_demand = Tensor(np.zeros((self.batch_s, 1, 1)))
        all_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape = (batch, problem+1, 1)
        self.data = torch.cat((all_node_xy, all_node_demand), dim=2)
        # shape = (batch, problem+1, 3)

    def reset(self, group_size):
        self.group_s = group_size
        self.group_state = GROUP_STATE(group_size=group_size, data=self.data)

        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = self.group_state.finished.all()  # state.finished.shape = (batch, group)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_travel_distance(self):
        all_node_xy = self.data[:, None, :, 0:2].expand(self.batch_s, self.group_s, -1, 2)
        # shape = (batch, group, problem+1, 2)
        gathering_index = self.group_state.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape = (batch, group, selected_count, 2)
        ordered_seq = all_node_xy.gather(dim=2, index=gathering_index)
        # shape = (batch, group, selected_count, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # size = (batch, group, selected_count)

        travel_distances = segment_lengths.sum(2)
        # size = (batch, group)
        return travel_distances


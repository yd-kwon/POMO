
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
def KNAPSACK_DATA_LOADER__RANDOM(num_sample, num_items, batch_size):
    dataset = KnapSack_Dataset__Random(num_sample=num_sample, num_items=num_items)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=knapsack_collate_fn)
    return data_loader


class KnapSack_Dataset__Random(Dataset):
    def __init__(self, num_sample, num_items):
        self.num_sample = num_sample
        self.num_items = num_items

    def __getitem__(self, index):
        data = np.random.rand(self.num_items, 2)
        return data

    def __len__(self):
        return self.num_sample


def knapsack_collate_fn(batch):
    return Tensor(batch)


####################################
# STATE
####################################
class STATE:

    def __init__(self, item_data, capacity):
        self.batch_s = item_data.size(0)
        self.items_and_a_dummy = Tensor(np.zeros((self.batch_s, PROBLEM_SIZE+1, 2)))
        self.items_and_a_dummy[:, :PROBLEM_SIZE, :] = item_data
        self.item_data = self.items_and_a_dummy[:, :PROBLEM_SIZE, :]
        # shape = (batch, problem, 2)

        # History
        ####################################
        self.current_node = None
        self.selected_count = 0
        self.selected_node_list = LongTensor(np.zeros((self.batch_s, 0)))
        # shape = (batch, selected_count)

        # Status
        ####################################
        self.accumulated_value = Tensor(np.zeros((self.batch_s,)))
        # shape = (batch,)
        self.capacity = Tensor(np.ones((self.batch_s,))) * capacity
        # shape = (batch,)
        self.ninf_mask_w_dummy = Tensor(np.zeros((self.batch_s, PROBLEM_SIZE+1)))
        self.ninf_mask = self.ninf_mask_w_dummy[:, :PROBLEM_SIZE]
        # shape = (batch, problem)
        self.fit_ninf_mask = None
        self.finished = BoolTensor(np.zeros((self.batch_s,)))
        # shape = (batch,)

    def move_to(self, selected_item_idx):
        # selected_item_idx.shape = (batch,)

        # History
        ####################################
        self.current_node = selected_item_idx
        self.selected_count += 1
        self.selected_node_list = torch.cat((self.selected_node_list, selected_item_idx[:, None]), dim=1)

        # Status
        ####################################
        gathering_index = selected_item_idx[:, None, None].expand(self.batch_s, 1, 2)
        selected_item = self.items_and_a_dummy.gather(dim=1, index=gathering_index).squeeze(dim=1)
        # shape = (batch, 2)

        self.accumulated_value += selected_item[:, 1]
        self.capacity -= selected_item[:, 0]

        self.ninf_mask_w_dummy[torch.arange(self.batch_s), selected_item_idx] = -np.inf
        unfit_bool = (self.capacity[:, None] - self.item_data[:, :, 0]) < 0
        # shape = (batch, problem)
        self.fit_ninf_mask = self.ninf_mask.clone()
        self.fit_ninf_mask[unfit_bool] = -np.inf

        self.finished = (self.fit_ninf_mask == -np.inf).all(dim=1)
        # shape = (batch,)
        self.fit_ninf_mask[self.finished[:, None].expand(self.batch_s, PROBLEM_SIZE)] = 0  # do not mask finished epi.


class GROUP_STATE:

    def __init__(self, group_size, item_data, capacity):
        self.batch_s = item_data.size(0)
        self.group_s = group_size
        self.items_and_a_dummy = Tensor(np.zeros((self.batch_s, PROBLEM_SIZE+1, 2)))
        self.items_and_a_dummy[:, :PROBLEM_SIZE, :] = item_data
        self.item_data = self.items_and_a_dummy[:, :PROBLEM_SIZE, :]
        # shape = (batch, problem, 2)

        # History
        ####################################
        self.current_node = None
        # shape = (batch_s, group)
        self.selected_count = 0
        self.selected_node_list = LongTensor(np.zeros((self.batch_s, self.group_s, 0)))
        # shape = (batch_s, group, selected_count)

        # Status
        ####################################
        self.accumulated_value = Tensor(np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)
        self.capacity = Tensor(np.ones((self.batch_s, self.group_s))) * capacity
        # shape = (batch, group)
        self.ninf_mask_w_dummy = Tensor(np.zeros((self.batch_s, self.group_s, PROBLEM_SIZE+1)))
        self.ninf_mask = self.ninf_mask_w_dummy[:, :, :PROBLEM_SIZE]
        # shape = (batch, group, problem)
        self.fit_ninf_mask = None
        self.finished = BoolTensor(np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)


    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # History
        ####################################
        self.current_node = selected_idx_mat
        self.selected_count += 1
        self.selected_node_list = torch.cat((self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)

        # Status
        ####################################
        items_mat = self.items_and_a_dummy[:, None, :, :].expand(self.batch_s, self.group_s, PROBLEM_SIZE+1, 2)
        gathering_index = selected_idx_mat[:, :, None, None].expand(self.batch_s, self.group_s, 1, 2)
        selected_item = items_mat.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape = (batch, group, 2)

        self.accumulated_value += selected_item[:, :, 1]
        self.capacity -= selected_item[:, :, 0]

        batch_idx_mat = torch.arange(self.batch_s)[:, None].expand(self.batch_s, self.group_s)
        group_idx_mat = torch.arange(self.group_s)[None, :].expand(self.batch_s, self.group_s)
        self.ninf_mask_w_dummy[batch_idx_mat, group_idx_mat, selected_idx_mat] = -np.inf

        unfit_bool = (self.capacity[:, :, None] - self.item_data[:, None, :, 0]) < 0
        # shape = (batch, group, problem)
        self.fit_ninf_mask = self.ninf_mask.clone()
        self.fit_ninf_mask[unfit_bool] = -np.inf

        self.finished = (self.fit_ninf_mask == -np.inf).all(dim=2)
        # shape = (batch, group)
        self.fit_ninf_mask[self.finished[:, :, None].expand(self.batch_s, self.group_s, PROBLEM_SIZE)] = 0
        # do not mask finished episode






####################################
# ENVIRONMENT
####################################
class ENVIRONMENT:

    def __init__(self, item_data):
        # item_data.shape = (batch, problem, 2)

        self.item_data = item_data
        self.batch_s = item_data.size(0)
        self.state = None

    def reset(self):
        if PROBLEM_SIZE == 50:
            capacity = 12.5
        elif PROBLEM_SIZE == 100:
            capacity = 25
        elif PROBLEM_SIZE == 200:
            capacity = 25
        else:
            raise NotImplementedError

        self.state = STATE(item_data=self.item_data, capacity=capacity)

        reward = None
        done = False
        return self.state, reward, done

    def step(self, selected_item_idx):
        # selected_node_idx.shape = (batch,)

        # move state
        self.state.move_to(selected_item_idx)

        # returning values
        done = self.state.finished.all()
        if done:
            reward = self.state.accumulated_value
        else:
            reward = None

        return self.state, reward, done


class GROUP_ENVIRONMENT:

    def __init__(self, item_data):
        # seq.shape = (batch, problem, 2)

        self.item_data = item_data
        self.batch_s = item_data.size(0)
        self.group_state = None

    def reset(self, group_size):
        if PROBLEM_SIZE == 50:
            capacity = 12.5
        elif PROBLEM_SIZE == 100:
            capacity = 25
        elif PROBLEM_SIZE == 200:
            capacity = 25
        else:
            raise NotImplementedError

        self.group_state = GROUP_STATE(group_size=group_size,
                                       item_data=self.item_data, capacity=capacity)

        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        reward = None
        done = self.group_state.finished.all()  # state.finished.shape = (batch, group)
        if done:
            reward = self.group_state.accumulated_value
        else:
            reward = None

        return self.group_state, reward, done


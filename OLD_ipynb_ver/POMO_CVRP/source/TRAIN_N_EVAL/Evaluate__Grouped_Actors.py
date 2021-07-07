
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


import numpy as np

# For debugging
from IPython.core.debugger import set_trace

# Hyper Parameters
from HYPER_PARAMS import *
from TORCH_OBJECTS import *

from source.utilities import Average_Meter, augment_xy_data_by_8_fold
from source.cvrp import CVRP_DATA_LOADER__RANDOM, GROUP_ENVIRONMENT


########################################
# EVAL
########################################

eval_result = []

def EVAL(grouped_actor, epoch, timer_start, logger):

    global eval_result

    grouped_actor.eval()

    eval_AM = Average_Meter()
    test_loader = CVRP_DATA_LOADER__RANDOM(num_sample=TEST_DATASET_SIZE,
                                           num_nodes=PROBLEM_SIZE,
                                           batch_size=TEST_BATCH_SIZE)

    with torch.no_grad():
        for depot_xy, node_xy, node_demand in test_loader:
            # depot_xy.shape = (batch, 1, 2)
            # node_xy.shape = (batch, problem, 2)
            # node_demand.shape = (batch, problem, 1)

            batch_s = depot_xy.size(0)

            env = GROUP_ENVIRONMENT(depot_xy, node_xy, node_demand)
            group_s = PROBLEM_SIZE
            group_state, reward, done = env.reset(group_size=group_s)
            grouped_actor.reset(group_state)

            # First Move is given
            first_action = LongTensor(np.zeros((batch_s, group_s)))  # start from node_0-depot
            group_state, reward, done = env.step(first_action)

            # Second Move is given
            second_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
            group_state, reward, done = env.step(second_action)

            while not done:
                action_probs = grouped_actor.get_action_probabilities(group_state)
                # shape = (batch, group, problem+1)
                action = action_probs.argmax(dim=2)
                # shape = (batch, group)
                action[group_state.finished] = 0  # stay at depot, if you are finished
                group_state, reward, done = env.step(action)

            max_reward, _ = reward.max(dim=1)
            eval_AM.push(-max_reward)  # reward was given as negative dist


    # LOGGING
    dist_avg = eval_AM.result()
    eval_result.append(dist_avg)


    logger.info('--------------------------------------------------------------------------')
    log_str = '  <<< EVAL after Epoch:{:03d} >>>   Avg.dist:{:f}'.format(epoch, dist_avg)
    logger.info(log_str)
    logger.info('--------------------------------------------------------------------------')
    logger.info('eval_result = {}'.format(eval_result))
    logger.info('--------------------------------------------------------------------------')
    logger.info('--------------------------------------------------------------------------')
    logger.info('--------------------------------------------------------------------------')


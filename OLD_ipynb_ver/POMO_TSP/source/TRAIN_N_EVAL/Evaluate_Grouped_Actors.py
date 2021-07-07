
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

from source.utilities import Average_Meter
from source.travelling_saleman_problem import TSP_DATA_LOADER__RANDOM, GROUP_ENVIRONMENT


########################################
# EVAL
########################################

eval_result = []

def update_eval_result(old_result):
    global eval_result
    eval_result = old_result

def EVAL(actor_group, epoch, timer_start, logger):

    global eval_result

    actor_group.eval()

    eval_dist_AM = Average_Meter()
    if TSP_SIZE == 5:
        raise NotImplementedError
    elif TSP_SIZE == 10:
        raise NotImplementedError
    else:
        test_loader = TSP_DATA_LOADER__RANDOM(num_sample=TEST_DATASET_SIZE, num_nodes=TSP_SIZE, batch_size=TEST_BATCH_SIZE)

    for data in test_loader:
        # data.shape = (batch_s, TSP_SIZE, 2)
        batch_s = data.size(0)

        with torch.no_grad():
            env = GROUP_ENVIRONMENT(data)
            group_s = TSP_SIZE
            group_state, reward, done = env.reset(group_size=group_s)
            actor_group.reset(group_state)

            # First Move is given
            first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
            group_state, reward, done = env.step(first_action)

            while not done:
                actor_group.update(group_state)
                action_probs = actor_group.get_action_probabilities()
                # shape = (batch, group, TSP_SIZE)
                action = action_probs.argmax(dim=2)
                # shape = (batch, group)
                group_state, reward, done = env.step(action)

        max_reward, _ = reward.max(dim=1)
        eval_dist_AM.push(-max_reward)  # reward was given as negative dist

    # LOGGING
    dist_avg = eval_dist_AM.result()
    eval_result.append(dist_avg)

    logger.info('--------------------------------------------------------------------------')
    log_str = '  <<< EVAL after Epoch:{:03d} >>>   Avg.dist:{:f}'.format(epoch, dist_avg)
    logger.info(log_str)
    logger.info('--------------------------------------------------------------------------')
    logger.info('eval_result = {}'.format(eval_result))
    logger.info('--------------------------------------------------------------------------')
    logger.info('--------------------------------------------------------------------------')
    logger.info('--------------------------------------------------------------------------')


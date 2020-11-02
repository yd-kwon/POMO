# POMO

This repository provides a reference implementation of *POMO* and saved trained models as described in the paper:<br>
> POMO: Policy Optimization with Multiple Optima for Reinforcement Learning<br>
> accepted at NeurIPS 2020<br>
http://arxiv.org/abs/2010.16011

The code is written using Pytorch.<br>
<br>

### Basic Usage

To test run, use an application (e.g. Jupyter Notebook) to open ipynb files.<br>
*Train.ipynb* contains codes for POMO training, which produces a model that you can save using torch.save()<br>
*Inference.ipynb* contains codes for inference using saved models.<br>
Examples of trained models are also provided in the folder named "result".<br>

You can edit *HYPER_PARAMS.py* to change the size of the problem or other hyper-parameters before training. <br>

Three example problems are solved:<br>
- Traveling Salesman Problem (TSP) <br>
- Capacitated Vehicle Routing Problem (CVRP) <br>
- 0-1 Knapsack Problem (KP) <br>

<br>

### Used Libraries
torch==1.2.0<br>
numpy==1.16.4<br>
ipython==7.1.1<br>
matplotlib==3.1.0<br>



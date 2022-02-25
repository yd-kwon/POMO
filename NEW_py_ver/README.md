
# POMO, as .py version

We provide codes for two CO (combinatorial optimization) problems:<br>
- Traveling Salesman Problem (TSP) <br>
- Capacitated Vehicle Routing Problem (CVRP) <br>


### Changes from the old version

Other than the re-organized structure, no major change has been made, so that the two versions should give roughly the same results.
Some meaningful changes are:
- Query token in the decoder does not contain "graph encoding" now, because this does not seem to make much difference. (But who knows?)
- Normalization methods have changed, from BatchNorm to InstanceNorm. (It seemed more natural. But this may have affected the model's performance in a negative way.) 


### Basic Usage

To train a model, run *train.py*. <br>
*train.py* contains parameters you can modify. At the moment, it is set to train N=20 problems. <br>
<br>
To test a model, run *test.py*. <br>
You can specify the model as a parameter contained in *test.py*. At the moment, it is set to use the saved model (N=20) we have provided (in *result* folder), but you can easily use the one you have trained using *train.py*.




import sys
import os
import numpy as np
import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.set_default_dtype(torch.float64)
debug._set_state(True)
# Get script directory
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(script_dir[:-12])

# Function network
from bofn.experiment_manager import experiment_manager
from bofn.utils.dag import DAG
from SIQR_utils import simulate,q_func_weight,q_func_scalar,L_simulate,runningMean
problem = 'SIQR_model_calibration'
time=30
populaion=100

def function_network_T(X) -> Tensor:
    state = {"I": 0.01, "S": 0.99, "R": 0., "Q": 0.}
    output = L_simulate(state,np.array(X),time,q_func_scalar)#q_func_weight
    return   torch.tensor(output.reshape(X.shape[0],4*time))

def function_network(X):
    state = {"I": 0.01, "S": 0.99, "R": 0., "Q": 0.}
    output =L_simulate(state,np.array(X),time,q_func_scalar)#q_func_weight
    return  torch.tensor(output.reshape(X.shape[0],4*time))

np.random.seed(0)
# setting 1: true underlying parameters
x0 = np.array([0.2,0.9,0.1,0.2]) #alpha,beta,gammaI,gammaQ
#setting 2
#x0 = np.array([0.3, 0.06, 0.12, 0.90, 0.1, 0.2])#  0.005*6 0.01*6 0.02*6

# observed values
mask= torch.ones(1,120)
#mask[0,torch.arange(1,120,4)]=0.
y_true= function_network_T(x0[None,:])
y_true_noise=np.clip(y_true+0.05* np.random.randn(*y_true.shape), 0., 1.0)

# Function that maps the network output to the objective value
obj_transform = lambda Y,X: -(  ((Y - y_true_noise)*populaion*mask) ** 2).mean(dim=-1)
obj_transform = GenericMCObjective(obj_transform)
obj_transform_true = lambda Y,X: -(  ((Y - y_true)*populaion) ** 2).mean(dim=-1)
obj_transform_true = GenericMCObjective(obj_transform_true)

parent_nodes = [] #i
for _ in range(4):
    parent_nodes.append([])#s
for t in range(time-1):
    parent_nodes.append([4*t,    4*t+1])           # i->i' s->i'
    parent_nodes.append([4*t+1])                   # s->s'
    parent_nodes.append([4*t+2,  4*t, 4*t+3])      # i->r' r->r', q->r',
    parent_nodes.append([4*t+3,  4*t,   ])         # i->q'   q->q'

dag = DAG(parent_nodes=parent_nodes)

# Active input indices I(X) for each nodes h
active_input_indices = []
for k in range(time):
    active_input_indices.append(list(range(len(x0))))
    active_input_indices.append(list(range(len(x0))))
    active_input_indices.append(list(range(len(x0))))
    active_input_indices.append(list(range(len(x0))))

# Run experiment
args={'n_init_evals':(2*len(x0) +1),
      'n_bo_iter':50,
      'restart':False,
      'function_network':function_network,
      'dag':dag,
      'input_dim':len(x0),
      'active_indices':active_input_indices,
      'obj_transform':obj_transform,
      'obj_transform_true': obj_transform_true,
      'bounds':torch.tensor([len(x0)*[0.],len(x0)*[1.]]),
      }
experiment_manager(problem=problem,algo='EICF',first_trial=1, last_trial=5,**args)



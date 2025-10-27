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
time=37
np.random.seed(0)
#torch.manual_seed(0)
import pandas as pd
y_true=pd.read_csv("covid_data.csv")['United Kingdom'].to_numpy()
y_true=runningMean(y_true[:,None],20).T
I_init=y_true.squeeze()[0]
y_true=torch.tensor(y_true)[...,torch.arange(0,365,10)]
y_true_noise=y_true
mask= torch.zeros(37*2,dtype=torch.bool)
mask[torch.arange(0,37*2,2)]=True

log_range_I, log_lower_I = -np.log(0.1), np.log(I_init) + np.log(0.1)
log_range_ppl, log_lower_ppl = 2*np.log(10), np.log(I_init) + np.log(10)
# Function that maps the network output to the objective value
def function_network(X):
    X=np.array(X)
    I   =np.exp(X[...,-2]*log_range_I + log_lower_I)
    ppl =np.exp(X[...,-1]*log_range_ppl + log_lower_ppl)
    state={'I':I/ppl,'S':(ppl-I)/ppl,'R':0.,'Q':0.}
    output =L_simulate(state,np.array(X[...,:-2]),365,q_func_scalar,step_size=0.1)[:,torch.arange(0,365,10),:]#q_func_weight
    #return  torch.tensor(output)[...,0]
    return torch.tensor(output[...,:2].reshape(X.shape[0], 2 * time))

def obj_transform(Y,X):
    scale = 1e-3  # -3 for uk -4 for us
    ppl=torch.exp(X[...,[-1]]*log_range_ppl + log_lower_ppl)
    loss=-(((Y[..., mask]*ppl - y_true_noise) * scale) ** 2).mean(dim=-1)
    return loss
obj_transform = GenericMCObjective(obj_transform)

def obj_transform_true(Y,X):
    scale = 1e-3  # -3 for uk
    ppl=torch.exp(X[...,[-1]]*log_range_ppl + log_lower_ppl)
    loss=-(((Y[..., mask]*ppl - y_true) * scale) ** 2).mean(dim=-1)
    return loss
obj_transform_true = GenericMCObjective(obj_transform_true)

parent_nodes = [] #i
for _ in range(2):
    parent_nodes.append([])#s

for t in range(time-1):
    parent_nodes.append([2*t, 2*t+1])           # i->i' s->i'
    parent_nodes.append([2*t+1])             # s->s'

dag = DAG(parent_nodes=parent_nodes)
# Active input indices I(X) for each nodes h
active_input_indices = []
for k in range(time):
    active_input_indices.append(list(range(6)))#list(range(8))) for q_func_weight
    active_input_indices.append(list(range(6)))
    active_input_indices.append(list(range(6)))
    active_input_indices.append(list(range(6)))

# Run experiment
args={'n_init_evals':(2*5 +1),
      'n_bo_iter':50,
      'restart':False,
      'function_network':function_network,
      'dag':dag,
      'input_dim':6,
      'active_indices':active_input_indices,
      'obj_transform':obj_transform,
      'obj_transform_true': obj_transform_true,
      'bounds': torch.tensor([[0.]*6,  [1.]*6]),
      }
experiment_manager(problem=problem,algo='KGCF',first_trial=1, last_trial=5,**args)



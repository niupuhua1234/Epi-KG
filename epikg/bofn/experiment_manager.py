from typing import Callable, List, Optional
#from bofn.bob_trial import bofn_trial
import os
import sys
import numpy as np
import torch
import time
#from bofn.utils.initial_design import generate_initial_design
from bofn.bofn_trial import get_new_suggested_point
from botorch.utils.sampling import draw_sobol_samples

def experiment_manager(
    problem: str,
    algo: str,
    first_trial: int, 
    last_trial: int,
    **args
) -> None:
    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/new/" + problem + "/" + algo + "/"

    if not os.path.exists(results_folder) : os.makedirs(results_folder)
    if not os.path.exists(results_folder + "runtimes/"): os.makedirs(results_folder + "runtimes/")
    if not os.path.exists(results_folder + "X/"):os.makedirs(results_folder + "X/")
    if not os.path.exists(results_folder + "Y/"): os.makedirs(results_folder + "Y/")
    if not os.path.exists(results_folder + "obj/"): os.makedirs(results_folder + "obj/")

    for trial in range(first_trial, last_trial + 1):
        bofn_trial(results_folder=results_folder,problem=problem,algo=algo,trial=trial,**args,)


def bofn_trial(
    results_folder:str,
    problem: str,
    algo: str,
    trial:int,
    n_init_evals: int,
    n_bo_iter: int,
    restart: bool,
    function_network: Callable,
    input_dim: int,
    obj_transform,
    obj_transform_true,
    bounds,
    **args
) -> None:
    if restart:
        # Check if training data is already available
        try:
            # Current available evaluations
            X = torch.tensor(np.loadtxt(results_folder + "X/X_" + str(trial) + ".txt"))
            Y = torch.tensor(np.loadtxt(results_folder + "Y/Y_" + str(trial) + ".txt"))
            obj = torch.tensor(np.loadtxt(results_folder + "obj/obj_" + str(trial) + ".txt"))
            # Historical best observed objective values and running times
            best_obj = list(np.loadtxt(results_folder + "best_obj_" + str(trial) + ".txt"))
            runtimes = list(np.loadtxt( results_folder + "runtimes/runtimes_" + str(trial) + ".txt"))
            init_iter = len(best_obj)
            print("Restarting experiment from available data.")
        except:
            raise TypeError("Fail to load historical data!")
    else:
        # uniformly initialization of X
        X = draw_sobol_samples(bounds=bounds, n=n_init_evals, q=1).squeeze(1)#generate_initial_design(num_samples=n_init_evals,input_dim=input_dim, seed=trial)
        Y = function_network(X)
        obj = obj_transform(Y,X)
        # Current best objective value
        #best_obj = [obj.max().item()]
        best_obj = [obj_transform_true(Y,X).max().item()]
        runtimes = []
        init_iter = 1

    # update by fitting a new gp-network or gps by [X_old,x] and [Y_old,y] and find the next condidate x'
    for iteration in range(init_iter, n_bo_iter + 1):
        #if best_obj[-1]<1e-4: break

        print("Experiment: " + problem)
        print("Sampling policy: " + algo)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # New suggested point
        t0 = time.time()
        X_new = get_new_suggested_point(algo=algo,
                                        X=X, Y=Y,obj=obj,
                                        bounds=bounds,
                                        obj_transform=obj_transform,**args,)
        t1 = time.time()
        runtimes.append(t1 - t0)
        Y_new = function_network(X_new) # Evalaute network at new point
        obj_new = obj_transform(Y_new,X_new) # Evaluate objective at new point
        # Update training data
        X = torch.cat([X, X_new], 0)
        Y = torch.cat([Y, Y_new], 0)
        obj = torch.cat([obj, obj_new], 0)
        # Update historical best observed objective values
        #best_obj.append( obj.max().item())
        best_obj.append(obj_transform_true(Y_new,X_new).item())

        print("Iteration run time: ",t1-t0,"s")
        print("Objective value found: " + str(best_obj[-1]))
        # Save data
        np.savetxt(results_folder + "X/X_" + str(trial) + ".txt", X.numpy(),fmt='%1.8f')
        np.savetxt(results_folder + "Y/Y_" + str(trial) + ".txt", Y.numpy(),fmt='%1.8f')
        np.savetxt(results_folder + "obj/obj_" +str(trial) + ".txt", obj.numpy(),fmt='%1.8f')
        np.savetxt(results_folder + "best_obj_" + str(trial) + ".txt", np.array(best_obj),fmt='%1.8f')
        np.savetxt(results_folder + "runtimes/runtimes_" + str(trial) + ".txt", np.array(runtimes),fmt='%1.8f')

            
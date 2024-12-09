import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from SIQR_epidemic_model_simulator import runningMean,q_func_scalar,q_func_weight,L_simulate,simulate

if __name__ == '__main__':
    #np.random.seed(0)
    import pandas as pd
    y_raw = pd.read_csv("covid_data.csv")['US'].to_numpy()[0:365]
    y_true = runningMean(y_raw[:, None], 20).T
    I_init = y_true.squeeze()[0]
    result = './real_us/SIQR_model_calibration_scalar/'
    a=[]
    for i in range(5):
        index=np.loadtxt(result+'KGCF/best_obj_{}.txt'.format(i+1))
        index=np.argmax(index)
        data=np.loadtxt(result+'KGCF/X/X_{}.txt'.format(i+1))
        x=data[10+index]#skip initial data

        I,ppl = np.exp(x[-2]),np.exp(x[-1])#np.exp(x[-2] * log_range_I + log_lower_I),np.exp(x[-1] * log_range_ppl + log_lower_ppl)
        state = {'I':I,'S':(ppl-I),'R':0.,'Q':0.}
        traj = L_simulate(state,x[:-2],365,q_func_scalar,step_size=0.1,ppl=ppl).squeeze()
        #plt.plot(traj[:, 0])
        a.append(traj)

    b=[]
    for i in range(5):
        index=np.loadtxt(result+'KGFN/best_obj_{}.txt'.format(i+1))
        index=np.argmax(index)
        data=np.loadtxt(result+'KGFN/X/X_{}.txt'.format(i+1))
        x=data[10+index]
        I,ppl = np.exp(x[-2]),np.exp(x[-1])#np.exp(x[-2] * log_range_I + log_lower_I),np.exp(x[-1] * log_range_ppl + log_lower_ppl)
        state = {'I':I,'S':(ppl-I),'R':0.,'Q':0.}
        traj = L_simulate(state,x[:-2],365,q_func_scalar,step_size=0.1,ppl=ppl).squeeze()
        #plt.plot(traj[:, 0])
        b.append(traj)

    time=np.arange(365)
    a=np.stack(a)
    a_mean=a.mean(0)
    a_std=a.std(0)

    b=np.stack(b)
    b_mean=b.mean(0)
    b_std=b.std(0)

    plt.figure(figsize=(6.4, 4.8))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4, -4))
    plt.plot(a_mean[:,0])
    plt.plot(b_mean[:, 0])
    plt.scatter(time,y_raw.squeeze(), color=  'gray',marker='o',alpha=0.2)
    plt.legend(['Calibration-KGCF','Calibration-KGFN','Observation'])
    plt.ylabel('Population',fontsize=20)
    plt.xticks([0, 92, 183, 274, 365],
               ["June", "Sep", "Dec", "March", "June"],fontsize=20)
    plt.ylim(-0.1 * 100000, 2.5 * 100000)

    plt.fill_between(time,a_mean[:,0]-a_std[:,0],a_mean[:,0]+a_std[:,0],alpha=0.2)
    plt.fill_between(time, b_mean[:, 0] - b_std[:, 0], b_mean[:, 0] + b_std[:, 0], alpha=0.2)
    plt.show()

    result = './real_us/SIQR_model_calibration_weight/'
    a = []
    for i in range(5):
        index = np.loadtxt(result + 'KGCF/best_obj_{}.txt'.format(i + 1))
        index = np.argmax(index)
        data = np.loadtxt(result + 'KGCF/X/X_{}.txt'.format(i + 1))
        x = data[16 + index]  # skip initial data

        I, ppl = np.exp(x[-2]), np.exp( x[-1])  # np.exp(x[-2] * log_range_I + log_lower_I),np.exp(x[-1] * log_range_ppl + log_lower_ppl)
        state = {'I': I, 'S': (ppl - I), 'R': 0., 'Q': 0.}
        traj = L_simulate(state, x[:-2], 365, q_func_weight, step_size=0.1, ppl=ppl).squeeze()
        # plt.plot(traj[:, 0])
        a.append(traj)

    b = []
    for i in range(5):
        index = np.loadtxt(result + 'KGFN/best_obj_{}.txt'.format(i + 1))
        index = np.argmax(index)
        data = np.loadtxt(result + 'KGFN/X/X_{}.txt'.format(i + 1))
        x = data[16 + index]
        I, ppl = np.exp(x[-2]), np.exp(x[-1])  # np.exp(x[-2] * log_range_I + log_lower_I),np.exp(x[-1] * log_range_ppl + log_lower_ppl)
        state = {'I': I, 'S': (ppl - I), 'R': 0., 'Q': 0.}
        traj = L_simulate(state, x[:-2], 365, q_func_weight, step_size=0.1, ppl=ppl).squeeze()
        # plt.plot(traj[:, 0])
        b.append(traj)

    time = np.arange(365)
    a = np.stack(a)
    a_mean = a.mean(0)
    a_std = a.std(0)

    b = np.stack(b)
    b_mean = b.mean(0)
    b_std = b.std(0)

    plt.figure(figsize=(6.4, 4.8))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4, -4))
    plt.plot(a_mean[:, 0])
    plt.plot(b_mean[:, 0])
    plt.scatter(time, y_raw.squeeze(), color='gray', marker='o', alpha=0.2)
    plt.legend(['Calibration-KGCF', 'Calibration-KGFN', 'Observation'])
    plt.ylabel('Population', fontsize=20)
    plt.xticks([0, 92, 183, 274, 365],
               ["June", "Sep", "Dec", "March", "June"], fontsize=20)
    plt.ylim(-0.1 * 100000, 2.5 * 100000)

    plt.fill_between(time, a_mean[:, 0] - a_std[:, 0], a_mean[:, 0] + a_std[:, 0], alpha=0.2)
    plt.fill_between(time, b_mean[:, 0] - b_std[:, 0], b_mean[:, 0] + b_std[:, 0], alpha=0.2)
    plt.pause()

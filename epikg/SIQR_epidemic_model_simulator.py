import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def q_func_scalar(state,params,ppl=1):
    alpha=params[...,0]
    return  alpha * state['I']
def q_func_weight(state,params,ppl=1):
    alpha_weight = params[...,0:-3]
    state_input  = np.stack([state['S']/ppl, state['I']/ppl, state['R']/ppl],axis=-1)
    #state_input = np.broadcast_to(np.stack([state['S'], state['I'], state['R']],axis=-1),alpha_weight.shape)
    alpha = np.log(np.einsum('...i,...i', alpha_weight, state_input) + 1.1)
    alphaI= alpha*state['I']
    return  alphaI


def node_ode(y,t,alpha,beta,gammaI,gammaQ,ppl):
    I,S,R,Q=y
    dydt   =[-gammaI*I + beta*S*I/ppl-alpha*I,
             -beta * S * I / ppl,
             gammaI * I+gammaQ*Q,
             alpha * I-gammaQ*Q
             ]
    return dydt

def node(state,params,h,q_func,ppl,t):
    #for siqr
    # Simulates the next state from the current state and the passed parameters (beta and gamma)
    beta, gamma_I, gamma_Q=params[...,-3],params[...,-2],params[...,-1]
    newstate = state.copy()
    #func_m = Memory(10., 1.0)
    betaI= state['S'] * beta*state['I']/ppl
    alphaI=q_func(state,params,ppl)
    gammaI= gamma_I*state['I']
    gammaQ = gamma_Q * state['Q']#(ppl-state['I']-state['S']-state['R'])

    newstate['I'] = state['I'] +h*(-gammaI + betaI-alphaI)
    newstate['S'] = state['S'] +h*(-betaI)
    newstate['R']=  state['R']+h*(gammaI+gammaQ)
    newstate['Q'] = state['Q'] +h*(alphaI-gammaQ)
    return newstate

from scipy.integrate import odeint

def simulate(state, params,T,q_func=q_func_scalar,step_size=1.0,ppl=1.):
    # Simulate forward in time, using node and the passed initial state and parameters
    params=params[None,:] if params.ndim <2 else params
    batch,dim=params.shape
    history = np.zeros([batch,T,len(state)])

    state['S']=np.broadcast_to(state['S'], params.shape[0])
    state['Q']=np.broadcast_to(state['Q'], params.shape[0])
    state['I']=np.broadcast_to(state['I'], params.shape[0])
    state['R']=np.broadcast_to(state['R'], params.shape[0])
    ppl=np.broadcast_to(ppl, params.shape[0])
    times=np.linspace(1, step_size * T, num=T)
    y0=np.stack(state.values())

    #initial state
    for batch,param_pl in enumerate(zip(params,ppl)):
        alpha, beta, gammaI, gammaQ=param_pl[0]
        pl=param_pl[1]
        history[batch,:, :]  = odeint(node_ode, list(y0[:,batch]), times,args=(alpha, beta,gammaI,gammaQ,pl))
    return history

def L_simulate(state, params,T,q_func=q_func_scalar,step_size=1.0,start_T=1,ppl=1.):
    # Simulate forward in time, using node and the passed initial state and parameters
    params=params[None,:] if params.ndim <2 else params
    batch,dim=params.shape
    history = np.zeros([batch,T,len(state)])

    state['S']=np.broadcast_to(state['S'], params.shape[0])
    state['Q']=np.broadcast_to(state['Q'], params.shape[0])
    state['I']=np.broadcast_to(state['I'], params.shape[0])
    state['R']=np.broadcast_to(state['R'], params.shape[0])
    #initial state
    for t in range(start_T):
        state = node(state,params, step_size,q_func,ppl,t)
    for t in range(T):
        state = node(state,params,step_size,q_func,ppl,t)
        history[:,t, :] = np.stack([*state.values()],axis=-1)  # history[t,:] has the state after
    return history

def runningMean(x, N):
    y = np.zeros_like(x)
    for ctr in range(len(x)):
        if ctr+1<N:
            y[ctr,:] = np.sum(x[0:ctr+1,:],0)+x[0,:]*(N-ctr-1)
        else:
            y[ctr,:]= np.sum(x[ctr+1-N:ctr+1,:],0)
    return y/N


if __name__ == '__main__':
    #np.random.seed(0)
    import pandas as pd
    y_raw = pd.read_csv("covid_data.csv")['United Kingdom'].to_numpy()
    y_true = runningMean(y_raw[:, None], 20).T
    I_init = y_true.squeeze()[0]
    log_range_I, log_lower_I = -np.log(0.1), np.log(I_init) + np.log(0.1)
    log_range_ppl, log_lower_ppl = 2*np.log(10), np.log(I_init) + np.log(10)

    result = 'C:/Users/niupu/bo/experiments/real_fnuk/SIQR_model_calibration/'
    b=[]
    for i in range(5):
        index=np.loadtxt(result+'KGCF/best_obj_{}.txt'.format(i+1))
        index=np.argmax(index)
        data=np.loadtxt(result+'KGCF/X/X_{}.txt'.format(i+1))
        x=data[10+index]

        I,ppl = np.exp(x[-2] * log_range_I + log_lower_I),np.exp(x[-1] * log_range_ppl + log_lower_ppl)
        state = {'I':I,'S':(ppl-I),'R':0.,'Q':0.}
        traj = L_simulate(state,x[:-2],365,q_func_scalar,step_size=0.1,ppl=ppl).squeeze()
        plt.plot(traj[:, 0])
        b.append(traj)

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
    plt.ylim(-0.2 * 10000, 5 * 10000)

    plt.fill_between(time,a_mean[:,0]-a_std[:,0],a_mean[:,0]+a_std[:,0],alpha=0.2)
    plt.fill_between(time, b_mean[:, 0] - b_std[:, 0], b_mean[:, 0] + b_std[:, 0], alpha=0.2)

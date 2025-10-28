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

def get_batch(true_y,time_length,data_size = 300,batch_time = 30,batch_size = 24):
  s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
  batch_y0 = true_y[s]  # (batch_size, 1, emb)
  batch_t = torch.linspace(1, batch_time , batch_time)*time_length/data_size
  batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (time, batch_size, 1, emb)
  return batch_y0, batch_t, batch_y#torch.normal(0.0,(0.1*batch_y))

def visualize(t,true_y, pred_y=None):
  fig,ax  = plt.subplots(2, 2, figsize=(7, 7))
  legend=['Infectious','Susceptible','Recoverd','Qurantined']

  for i in range(4):
    ax[i//2][i%2].plot(t, true_y[:, i])
    ax[i//2][i%2].plot(t, pred_y[:, i])
    ax[i // 2][i % 2].legend(['True','Predicted'])
    ax[i // 2][i % 2].set_title(legend[i])
    ax[i//2][i%2].set_xlabel('Days',fontsize=5)
    ax[i // 2][i % 2].set_ylabel('Rate',fontsize=5)
    ax[i // 2][i % 2].set_ylim(0, 1)
  plt.show()



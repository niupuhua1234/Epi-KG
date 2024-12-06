import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
"""
This code runs a SI (no R) epidemiological model with n=1 groups. Each node is a time period, in which infections occur.

Consider a time period t and use I[i] to indicate the fraction of population i infected at the start of this time period.
Then,
SIQR:
S(t+1)-S(t) =  -β*I(t)*S(t)/N  
I(t+1)-I(t) =  β*I(t)*S(t)/N -γ[I]*I(t)-α*I(t)
R(t+1)-R(t) =  γ[I]*I(t)+γ[Q]*Q(t)
Q(t+1)-Q(t) =  αI(t)-γ[Q]*Q(t)

SEIR:
S(t+1)-S(t) =  -β*I(t)*S(t)/N  
I(t+1)-I(t) =  κ[Q]*E(t)-γ[I]*I(t)
R(t+1)-R(t) =  γ[I]*I(t)
E(t+1)-E(t) =  β*I(t)*S(t)/N - κ*E(t)


Each infectious person in group j at the beginning of time period t comes in to contact with beta_ij[t] people from group i.
A fraction S(t) of these people are susceptible. Thus, the number of people infected on this time period is 
(N * I(t)) * β * S(t)

In addition, at the start of the period, a fraction I(t) was infected. 
Also, a fraction γI of the infected people recovered, resulting in a decrease of  γ[I] * I(t) in the fraction infected.  
Beside, a  fraction of α*I(t) is move to be quarantined. 

Putting this all together, the new value of I(t+1) at the end of time period t is (1- γ[I])I(t)+ β*S(t)*I(t)-α*I(t)
"""

def uninterleave(lst):
    return lst[::2], lst[1::2]
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
def make_layer(activation,liner=False):
    def layer(W,b,h):
        return activation(np.einsum('kji,ki->kj',W.reshape(b.shape+(-1,)),h)+b)
    return layer
sigmoid_layer = make_layer(sigmoid)
linear_layer = make_layer(lambda x: x)

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

dims=[3*3,3,3*1,1]
dims_idx=[reduce(lambda x, y: x + y, dims[:i]) for i in range(1,len(dims))]
alpha_q=[]
def q_func_neural(state,params,ppl=1):
    alpha_params=uninterleave(np.split(params[...,0:-3],dims_idx,axis=-1))
    alpha = np.stack([state['S']/ppl, state['I']/ppl, state['R']/ppl],axis=-1)
    #alpha =  np.broadcast_to(np.stack([state['S'], state['I'], state['R']],axis=-1), (params.shape[0],3))
    for W,b in  alpha_params:
        alpha=sigmoid_layer(W,b,alpha)
    alphaI=alpha.squeeze()*state['I']
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
    newstate['S'] = state['S'] +h*(-betaI)#+ func_m(torch.tensor(t/10)).detach().numpy()/100
    newstate['R']=  state['R']+h*(gammaI+gammaQ)#- func_m(torch.tensor(t/10)).detach().numpy()/100
    newstate['Q'] = state['Q'] +h*(alphaI-gammaQ)
    return newstate


def node_seir(state,params,h,q_func,ppl):
    # Simulates the next state from the current state and the passed parameters (beta and gamma)
    beta, gamma, kappa=params[...,-3],params[...,-2],params[...,-1]
    newstate = state.copy()
    #for i in range(len(state)):  # number of state= number of population
    betaI= state['S'] * beta*state['I']/ppl
    gammaI= gamma*state['I']
    kappaQ= kappa * state['Q']

    newstate['I'] = state['I'] +h*( kappaQ-gammaI )
    newstate['S'] = state['S'] +h*(-betaI)
    newstate['R']=  state['R']+h* gammaI
    newstate['Q'] = state['Q'] +h*(-kappaQ+ betaI)
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
# S_stable =(alpha+gammaI)/beta
# Q_stable=alpha/gammaQ *I_stable
if __name__ == '__main__':
    #np.random.seed(0)
    import pandas as pd
    y_raw = pd.read_csv("covid_data.csv")['United Kingdom'].to_numpy()[92:457]#[92:457]
    y_true = runningMean(y_raw[:, None], 20).T
    I_init = y_true.squeeze()[0]
    log_range_I, log_lower_I = -np.log(0.1), np.log(I_init) + np.log(0.1)
    log_range_ppl, log_lower_ppl = 2*np.log(10), np.log(I_init) + np.log(10)

    case='scalar'
    if case=='scalar':
        x = np.array([0.2,0.9,0.1,0.2])# alpha,beta,gamma I,gammaQ [0.2 0.15*6 0.013*6, 0.1*6]
        x=np.array([0.12919033, 0.58201833, 0.14793122, 0.00000000, 0.00000000, 0.68479070])

        I,ppl = np.exp(x[-2] * log_range_I + log_lower_I),np.exp(x[-1] * log_range_ppl + log_lower_ppl)
        #state = {"I":0.01,"S":0.99,"R":0.,"Q":0.}
        state = {'I':I/ppl,'S':(ppl-I)/ppl,'R':0.,'Q':0.}
        history0 = L_simulate(state,x[:-2],365,q_func_scalar,step_size=0.1).squeeze()
        plt.plot(y_true.squeeze() / ppl)
        plt.plot(history0[:,0])
        plt.plot(history0[:,1])
        plt.plot(history0[:,2])
        plt.plot(history0[:,3])
        plt.legend(['Infectious','Susceptible','Recoverd','Qurantined'])
        plt.title('EI-CF')
        plt.xlabel('Days',fontsize=24)
        plt.ylabel('Rate',fontsize=24)

    elif case == 'weight':
        x = np.array([0.3, 0.06, 0.12, 0.90, 0.1, 0.2]) # 0.005*6 0.01*6 0.02*6
        x = np.array([0.16434699, 0.68829876, 0.55357996, 0.49904158, 0.01751214, 0.69670288, 6.73458823, 12.60660665])

        state = {'I': np.exp(x[-2])/np.exp(x[-1]), 'S': (np.exp(x[-1]) - np.exp(x[-2]))/np.exp(x[-1]), 'R': 0., 'Q': 0.}
        history0 = L_simulate(state, x[:-2], 365, q_func_weight, step_size=0.1).squeeze()
        plt.plot(y_true.squeeze() / np.exp(x[-1]))
        plt.plot(history0[:,0])
        plt.plot(history0[:,1])
        plt.plot(history0[:,2])
        plt.plot(history0[:,3])
        plt.legend(['Infectious','Susceptible','Recoverd','Qurantined'])
        plt.title('EI-CF')
        plt.xlabel('Days',fontsize=20)
        plt.ylabel('Rate',fontsize=20)

    elif case == 'neural':
        rng = np.random.default_rng(12345)
        #x = np.concatenate((rng.normal(size=51),np.array([ 0.90, 0.1, 0.2]))) # 0.005*6 0.01*6 0.02*6
        x  = np.array([0.64830024, 0.88409332, 1.00000000, 0.21554542, 0.00000000,
                       0.15376841, 0.00000000, 0.81428368, 0.00000000, 0.00000000,
                       0.00000000, 0.00000000, 0.44439538, 0.24842395, 0.93305142,
                       0.00000000, 0.66041464, 0.00000000, 0.83271183, 14.59235530])
        state = {'I': I_init/np.exp(x[-1]), 'S': (np.exp(x[-1]) - I_init)/np.exp(x[-1]), 'R': 0., 'Q': 0.}
        history0 = L_simulate(state, x[:-1],365, q_func_neural, step_size=0.1).squeeze()
        plt.plot(history0[:,0])
        plt.plot(history0[:,1])
        plt.plot(history0[:,2])
        plt.plot(history0[:,3])
        plt.legend(['Infectious','Susceptible','Recoverd','Qurantined'])
        plt.title('EI-CF')
        plt.xlabel('Days',fontsize=20)
        plt.ylabel('Rate',fontsize=20)



    result = 'C:/Users/niupu/bo/experiments/real_uk/SIQR_model_calibration/'
    a=[]
    for i in range(5):
        index=np.loadtxt(result+'EICF/best_obj_{}.txt'.format(i+1))
        index=np.argmax(index)
        data=np.loadtxt(result+'EICF/X/X_{}.txt'.format(i+1))
        x=data[10+index]# skip the initial parameter set

        state = {'I':   np.exp(x[-2]), 'S': np.exp(x[-1]) - np.exp(x[-2]) , 'R': 0., 'Q': 0.}
        traj = L_simulate(state,x[:-2],365,q_func_scalar,step_size=0.1,ppl=np.exp(x[-1]) ).squeeze()
        plt.plot(traj[:, 0])
        a.append(traj)

    result = 'C:/Users/niupu/bo/experiments/real_fnuk/SIQR_model_calibration/'
    b=[]
    for i in range(5):
        index=np.loadtxt(result+'EIFN/best_obj_{}.txt'.format(i+1))
        index=np.argmax(index)
        data=np.loadtxt(result+'EIFN/X/X_{}.txt'.format(i+1))
        x=data[10+index]

        I,ppl = np.exp(x[-2] * log_range_I + log_lower_I),np.exp(x[-1] * log_range_ppl + log_lower_ppl)
        state = {'I':I,'S':(ppl-I),'R':0.,'Q':0.}
        #state = {'I':   np.exp(x[-2]), 'S': np.exp(x[-1]) - np.exp(x[-2]) , 'R': 0., 'Q': 0.}
        traj = L_simulate(state,x[:-2],365,q_func_scalar,step_size=0.1,ppl=ppl).squeeze()
        plt.plot(traj[:, 0])
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
    # plt.plot(a_mean[:,1])
    # plt.plot(a_mean[:,2])
    # plt.plot(a_mean[:,3])
    #plt.legend(['Infectious','Susceptible','Recoverd','Qurantined'])
    #plt.xlabel('Day',fontsize=20)
    plt.legend(['Calibration-KGCF','Calibration-KGFN','Observation'])
    plt.ylabel('Population',fontsize=20)
    plt.xticks([0, 92, 183, 274, 365],
               ["June", "Sep", "Dec", "March", "June"],fontsize=20)
    #plt.ylim(-0.1 * 100000, 2.5 * 100000)
    plt.ylim(-0.2 * 10000, 5 * 10000)

    plt.fill_between(time,a_mean[:,0]-a_std[:,0],a_mean[:,0]+a_std[:,0],alpha=0.2)
    plt.fill_between(time, b_mean[:, 0] - b_std[:, 0], b_mean[:, 0] + b_std[:, 0], alpha=0.2)
    # plt.fill_between(time,a_mean[:,1]-a_std[:,1],a_mean[:,1]+a_std[:,1],alpha=0.2)
    # plt.fill_between(time,a_mean[:,2]-a_std[:,2],a_mean[:,2]+a_std[:,2],alpha=0.2)
    # plt.fill_between(time,a_mean[:,3]-a_std[:,3],a_mean[:,3]+a_std[:,3],alpha=0.2)

    # history0 = simulate(state,np.array([0.2,0.90,0.1,0.2] ), 30, q_func_scalar, 1.0).squeeze()
    # plt.plot(history0[:,0],color='#1f77b4',linestyle='--')
    # plt.plot(history0[:, 1], color= '#ff7f0e',linestyle='-.')
    # plt.plot(history0[:, 2], color= '#2ca02c',linestyle='-.')
    # plt.plot(history0[:, 3], color='#d62728',linestyle='-.')


    def max_seq(x):
        for j in range(0, len(x)):
            for i in range(1, len(x[0])):
                if x[j][i - 1] > x[j][i]:
                    x[j][i] = x[j][i - 1]
        return x
    #C:\Users\niupu\Downloads\DGCF
    result = 'C:/Users/niupu/bo/experiments/partial_results/SIQR_model_calibration/'
    a = []
    a.append(np.loadtxt(result + 'EI/best_obj_1.txt'))
    a.append(np.loadtxt(result + 'EI/best_obj_2.txt'))
    a.append(np.loadtxt(result + 'EI/best_obj_3.txt'))
    a.append(np.loadtxt(result + 'EI/best_obj_4.txt'))
    a.append(np.loadtxt(result + 'EI/best_obj_5.txt'))
    a=max_seq(a)

    b=[]
    b.append(np.loadtxt(result+'KG/best_obj_1.txt'))
    b.append(np.loadtxt(result+'KG/best_obj_2.txt'))
    b.append(np.loadtxt(result+'KG/best_obj_3.txt'))
    b.append(np.loadtxt(result+'KG/best_obj_4.txt'))
    b.append(np.loadtxt(result+'KG/best_obj_5.txt'))
    b=max_seq(b)
    #
    c=[]
    c.append(np.loadtxt(result+'EICF/best_obj_1.txt'))
    c.append(np.loadtxt(result+'EICF/best_obj_2.txt'))
    c.append(np.loadtxt(result+'EICF/best_obj_3.txt'))
    c.append(np.loadtxt(result+'EICF/best_obj_4.txt'))
    c.append(np.loadtxt(result+'EICF/best_obj_5.txt'))

    c=max_seq(c)


    # d=[]
    # d.append(np.loadtxt(result+'KGCF/best_obj_1.txt'))
    # d.append(np.loadtxt(result+'KGCF/best_obj_2.txt'))
    # d.append(np.loadtxt(result+'KGCF/best_obj_3.txt'))
    # d.append(np.loadtxt(result+'KGCF/best_obj_4.txt'))
    # d.append(np.loadtxt(result+'KGCF/best_obj_5.txt'))
    # d=max_seq(d)


    e=[]
    e.append(np.loadtxt(result+'EIFN/best_obj_1.txt'))
    e.append(np.loadtxt(result+'EIFN/best_obj_2.txt'))
    e.append(np.loadtxt(result+'EIFN/best_obj_3.txt'))
    e.append(np.loadtxt(result+'EIFN/best_obj_4.txt'))
    e.append(np.loadtxt(result+'EIFN/best_obj_5.txt'))
    e=max_seq(e)



    f=[]
    f.append(np.loadtxt(result+'DGCF/best_obj_1.txt'))
    f.append(np.loadtxt(result+'DGCF/best_obj_2.txt'))
    f.append(np.loadtxt(result+'DGCF/best_obj_3.txt'))
    f.append(np.loadtxt(result+'DGCF/best_obj_4.txt'))
    f.append(np.loadtxt(result+'DGCF/best_obj_5.txt'))
    f=max_seq(f)


    # plt.figure(figsize=(7,4))
    # plt.plot(-np.stack(a).mean(0))
    # plt.plot(-np.stack(b).mean(0))
    # plt.plot(-np.stack(c).mean(0))
    # #plt.plot(-np.stack(d).mean(0))
    # plt.plot(-np.stack(e).mean(0))
    # plt.plot(-np.stack(f).mean(0))
    #
    # plt.fill_between(np.arange(51),-np.stack(a).mean(0)-np.stack(a).std(0),-np.stack(a).mean(0)+np.stack(a).std(0),alpha=0.2)
    # plt.fill_between(np.arange(51),-np.stack(b).mean(0)-np.stack(b).std(0),-np.stack(b).mean(0)+np.stack(b).std(0),alpha=0.2)
    # plt.fill_between(np.arange(51),-np.stack(c).mean(0)-np.stack(c).std(0),-np.stack(c).mean(0)+np.stack(c).std(0),alpha=0.2)
    # #plt.fill_between(np.arange(51),-np.stack(d).mean(0)-np.stack(d).std(0),-np.stack(d).mean(0)+np.stack(d).std(0),alpha=0.2)
    # plt.fill_between(np.arange(51),-np.stack(e).mean(0)-np.stack(e).std(0),-np.stack(e).mean(0)+np.stack(e).std(0),alpha=0.2)
    # plt.fill_between(np.arange(51),-np.stack(f).mean(0)-np.stack(f).std(0),-np.stack(f).mean(0)+np.stack(f).std(0),alpha=0.2)
    #
    # plt.legend(['EI','KG','KGCF','KGFN','DGCF'])
    # plt.title('MSE')
    # plt.xlim(15,50)
    # plt.ylim(-10,100)
    # plt.show()
    #
    plt.figure(figsize=(6.4,4.8))
    plt.plot(np.log10(-np.stack(a)).mean(0))
    plt.plot(np.log10(-np.stack(b)).mean(0))
    plt.plot(np.log10(-np.stack(c)).mean(0))
    #plt.plot(np.log10(-np.stack(d)).mean(0))
    plt.plot(np.log10(-np.stack(e)).mean(0))
    plt.plot(np.log10(-np.stack(f)).mean(0))

    plt.fill_between(np.arange(51),np.log10(-np.stack(a)).mean(0)-np.log10(-np.stack(a)).std(0),
                     np.log10(-np.stack(a)).mean(0)+np.log10(-np.stack(a)).std(0),alpha=0.2)
    plt.fill_between(np.arange(51),np.log10(-np.stack(b)).mean(0)-np.log10(-np.stack(b)).std(0),
                     np.log10(-np.stack(b)).mean(0)+np.log10(-np.stack(b)).std(0),alpha=0.2)
    plt.fill_between(np.arange(51),np.log10(-np.stack(c)).mean(0)-np.log10(-np.stack(c)).std(0),
                     np.log10(-np.stack(c)).mean(0)+np.log10(-np.stack(c)).std(0),alpha=0.2)
    #plt.fill_between(np.arange(51),np.log10(-np.stack(d)).mean(0)-np.log10(-np.stack(d)).std(0),
    #                 np.log10(-np.stack(d)).mean(0)+np.log10(-np.stack(d)).std(0),alpha=0.2)
    plt.fill_between(np.arange(51),np.log10(-np.stack(e)).mean(0)-np.log10(-np.stack(e)).std(0),
                     np.log10(-np.stack(e)).mean(0)+np.log10(-np.stack(e)).std(0),alpha=0.2)
    plt.fill_between(np.arange(51),np.log10(-np.stack(f)).mean(0)-np.log10(-np.stack(f)).std(0),
                     np.log10(-np.stack(f)).mean(0)+np.log10(-np.stack(f)).std(0),alpha=0.2)

    plt.legend(['EI','KG','KGCF','KGFN','DGCF'])

    plt.title('log-MSE')
    plt.show()

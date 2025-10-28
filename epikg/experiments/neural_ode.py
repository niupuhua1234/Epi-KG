# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from   torchdiffeq import  odeint_adjoint,odeint
from  tqdm import  tqdm
import time
from SIQR_utils import  get_batch,L_simulate,runningMean,visualize
from ODE_base import NNnet,UDE
torch.manual_seed(0)


import pandas as pd
true_real=pd.read_csv("covid_data.csv")['US'].to_numpy()[0:365]
true_real=runningMean(true_real[:,None],20)
I_init=true_real.squeeze()[0]
path="./real_us/SIQR_model_calibration_scalar/KGCF/"
trail='4'
index= np.argmax(np.loadtxt(path+"best_obj_"+trail+".txt"))
x    = np.loadtxt(path+"X/X_"+trail+".txt")
x    = x[index+10]#skip initial x

#log_range_I, log_lower_I = -np.log(0.1), np.log(I_init) + np.log(0.1)
#log_range_ppl, log_lower_ppl =  2*np.log(10), np.log(I_init) + np.log(10)
#I, ppl = np.exp(x[-2] * log_range_I + log_lower_I), np.exp(x[-1] * log_range_ppl + log_lower_ppl)
I,ppl  =np.exp(x[-2]),np.exp(x[-1])
state = {'I': I/ ppl,'S': (ppl - I) / ppl, 'R': 0., 'Q': 0.}
##
time_length = 36.5
true_y =torch.tensor(L_simulate(state,x[:-2],365,step_size=0.1),dtype=torch.float).squeeze()
t = torch.linspace(1, 36.5, 365)
# true_y[:,-1]=(true_y[:,0]+true_y[:,-1])
true_y[:,0]=torch.tensor(true_real.squeeze()/ppl)
# true_y[:,-1]=true_y[:,-1]-true_y[:,0]
true_y0 = true_y[0]
##Train
time_size = 365
batch_time = 30
batch_size = 24
niters = 2000
torch.manual_seed(0)
UDE=UDE(x,NNnet(3,1,hidden_dim=20,n_hidden_layers=2))
optimizer = optim.Adam(UDE.parameters(), lr=5e-4)
start_time = time.time()

losses=[]
loss_best=np.inf
for iter in (pbar:=tqdm(range(niters + 1))):
  optimizer.zero_grad()
  batch_y0, batch_t, batch_y = get_batch(true_y,time_length,time_size,batch_time,batch_size)
  pred_y = odeint_adjoint(UDE,batch_y0,batch_t,method='euler')#='dopri5'
  loss = torch.mean(torch.square(pred_y[:,:,[0,3]] - batch_y[:,:,[0,3]]))
  loss.backward()
  optimizer.step()
  losses.append(loss.detach().item())

  if iter % 10 == 0 and iter!=0:
    with torch.no_grad():
      pred_y =odeint(UDE, true_y0.unsqueeze(0), torch.linspace(1,time_length,time_size) ,method='euler')# odeint_adjoint(UDE, true_y0.unsqueeze(0), t,method='dopri5')
      loss =torch.mean(torch.abs(pred_y[:,:,0]-true_y.unsqueeze(1)[:,:,0]))
      pbar.set_postfix({'Total Loss': '{:.6f}'.format(loss.item())})
      #if iter % 500 == 0 and iter != 0:
      #  visualize( torch.linspace(1,time_length,time_size), true_y.numpy(), pred_y[:,0,:].detach().numpy())
      if loss<loss_best:
        loss_best=loss
        np.savetxt('us_pred_y_'+trail+'.txt',pred_y[:,0,0]*ppl)

end_time = time.time() - start_time

print('process time: {} sec'.format(end_time))


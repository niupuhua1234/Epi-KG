import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# define dynamic function
class NNnet(nn.Module):
  def __init__(self,input_dim, output_dim,
               hidden_dim=124,n_hidden_layers=2):
    super().__init__()
    self.net = [nn.Linear(input_dim, hidden_dim),nn.Tanh()]
    for _ in range(n_hidden_layers - 1):
      self.net.append(nn.Linear(hidden_dim, hidden_dim))
      self.net.append(nn.Tanh())
    self.net.append(nn.Linear(hidden_dim,output_dim))
    self.net.append(nn.Sigmoid())
    self.net=nn.Sequential(*self.net)

    for m in self.net.modules():
      if isinstance(m, nn.Linear):
        #nn.init.normal_(m.weight, mean=0, std=2.0)
        #nn.init.constant_(m.bias, val=0)
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

  def forward(self, x):
    output =  self.net(x)
    return output.squeeze()

class UDE(nn.Module):
  def __init__(self,x,NNnet,embed_dim=32,time_ratio=0.1):
    super(UDE,self).__init__()
    self.NNnet=NNnet

    self.beta =Parameter(torch.tensor(x[1],dtype=torch.float))
    self.gammaI= Parameter(torch.tensor(x[2],dtype=torch.float))
    self.gammaQ=  Parameter(torch.tensor(x[3],dtype=torch.float))
    self.embed_dim=embed_dim
    self.time_ratio=time_ratio


  def forward(self, t, y):
    #    S(t + 1) - S(t) = -β * I(t) * S(t)
    #    I(t + 1) - I(t) = β * I(t) * S(t) - γ[I] * I(t) - α * I(t)
    #    R(t + 1) - R(t) = γ[I] * I(t) + γ[Q] * Q(t)
    #    Q(t + 1) - Q(t) = αI(t) - γ[Q] * Q(t)
    betaI= self.beta*y[:,0]*y[:,1]
    alphaI=self.NNnet(y[:,0:-1])*y[:,0]

    gammaII=self.gammaI*y[:,0]
    gammaQQ=self.gammaQ*y[:,-1]
    ############################
    dI= betaI-gammaII-alphaI
    dS=-betaI
    dR= gammaII+ gammaQQ
    dQ=-gammaQQ+alphaI
    return torch.stack([dI,dS,dR,dQ],dim=-1)


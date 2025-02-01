#! /usr/bin/env python3

r"""
Gaussian Process Network.
"""
from __future__ import annotations
import torch
from typing import Any,Optional,List
from botorch.models.model import Model
from botorch.models.model import FantasizeMixin
#from botorch.models import FixedNoiseGP
from botorch.models.gp_regression import SingleTaskGP,FixedNoiseGP
from botorch import fit_gpytorch_model
from botorch.posteriors import Posterior
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from bofn.utils.dag import DAG
from botorch.models.utils.assorted import fantasize as fantasize_flag
from botorch import settings

from torch import Tensor
import time
#fix_GP   train_X,      (batch=10,ndata(n)=12,ndim(d)=3)   10 indepent GP joint    with kernal = 12*3*3*12=12*12,
# fix GP.posterior      (batch=10,      n=12,       d=3)   10 indepent GP posteriror distriution
#  batch_dim can not be changed for the downstream optimization,  while n is growing.

# EI-CF                Model(.fantasy)->(BatchGPyTorchModel->PyTorchModel(.posterior)->(single_task_)GP)
#def posterior:  GP.posterior-> return GPyTorchPosterior
#def condition_observation GP.condition_on_observations-> return GP
#def fantasy:    GP.fantasy  ->return GP.condition_on_observations

#EI-FN                Model-> GP-network (.posterior)
#def posterior:  GP-network.posterior-> Multivarite_network->GP.posterior-> return GPyTorchPosterior
#def condition_observation GP-network.condition_on_observations-> GP.condition_on_observations->return GP
#def fantasy:    GP.fantasy  ->return GP-network.condition_on_observations


class GaussianProcessNetwork(Model):
    def __init__(self, train_X, train_Y,dag, active_indices,
                 train_Yvar=None, node_GPs:List[SingleTaskGP]=None,
                 Z_lr=None, Z_up=None)-> None:
        super(Model, self).__init__()  ####
        self.train_X = train_X
        self.train_Y = train_Y
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.active_indices = active_indices
        self.train_Yvar = train_Yvar


        if node_GPs is not None:
            self.node_GPs:List[ SingleTaskGP] = node_GPs
            self.Z_lr = Z_lr # for parents input/sample normizalition to (0,1)
            self.Z_up = Z_up
        else:
            self.node_GPs:List[ SingleTaskGP|None] = [None for k in range(self.n_nodes)]
            self.node_mlls:List[ExactMarginalLogLikelihood|None] = [None for k in range(self.n_nodes)]
            self.Z_lr:List[torch.Tensor|None] = [[None for j in range(len(self.dag.get_parent_nodes(k)))] for k in range(self.n_nodes)]
            self.Z_up:List[torch.Tensor|None] = [[None for j in range(len(self.dag.get_parent_nodes(k)))] for k in range(self.n_nodes)]

            for k in self.root_nodes:
                if self.active_indices is not None:
                    train_X_node_k = train_X[..., self.active_indices[k]]
                else:
                    train_X_node_k = train_X
                train_Y_node_k = train_Y[..., [k]]
                #self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))
                self.node_GPs[k] = FixedNoiseGP(train_X=train_X_node_k, train_Y=train_Y_node_k,
                                                train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-6,
                                                outcome_transform=Standardize(m=1,min_stdv=1e-20))
                #fit the GP for each node
                self.node_mlls[k] = ExactMarginalLogLikelihood(self.node_GPs[k].likelihood, self.node_GPs[k])
                fit_gpytorch_model(self.node_mlls[k])

            for k in range(self.n_nodes):
                if self.node_GPs[k] is None:
                    aux = train_Y[..., self.dag.get_parent_nodes(k)].clone() # h of parent nodes
                    for j in range(len(self.dag.get_parent_nodes(k))):
                        self.Z_lr[k][j] = torch.tensor(0.)#torch.min(aux[..., j])
                        self.Z_up[k][j] = torch.tensor(1.)#torch.max(aux[..., j])
                        aux[..., j] = (aux[..., j] - self.Z_lr[k][j])/(self.Z_up[k][j] - self.Z_lr[k][j])
                    train_X_node_k = torch.cat([train_X[..., self.active_indices[k]], aux], -1)
                    train_Y_node_k = train_Y[..., [k]]
                    self.node_GPs[k] = FixedNoiseGP(train_X=train_X_node_k, train_Y=train_Y_node_k,
                                                    train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-6,
                                                    outcome_transform=Standardize(m=1,min_stdv=1e-20,batch_shape=torch.Size([])))
                    self.node_mlls[k] = ExactMarginalLogLikelihood(self.node_GPs[k].likelihood, self.node_GPs[k])
                    fit_gpytorch_model(self.node_mlls[k])
        print('GP fitting Completed')

    def posterior(self, X: Tensor, observation_noise=False,**kwargs) -> MultivariateNormalNetwork:
        r"""Computes the posterior over model outputs at the provided points.
        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).
        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        return MultivariateNormalNetwork(self.node_GPs, self.dag, X, self.active_indices, self.Z_lr, self.Z_up)

    def forward(self, X: Tensor) -> MultivariateNormalNetwork:
        return MultivariateNormalNetwork(self.node_GPs, self.dag, X, self.active_indices, self.normalization_constant)


    def fantasize(self,
                  X: Tensor,
                  sampler,
        observation_noise: Optional[Tensor] = None,
        **kwargs: Any,
    )->Model:
        r"""Construct a fantasy model.

        Constructs a fantasy model in the following fashion:
        (1) compute the model posterior at `X`, including observation noise.
        If `observation_noise` is a Tensor, use it directly as the observation
        noise to add.
        (2) sample from this posterior (using `sampler`) to generate "fake"
        observations.
        (3) condition the model on the new fake observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            sampler: The sampler used for sampling from the posterior at `X`.
            observation_noise: A `model_batch_shape x 1 x m`-dim tensor or
                a `model_batch_shape x n' x m`-dim tensor containing the average
                noise for each batch and output, where `m` is the number of outputs.
                `noise` must be in the outcome-transformed space if an outcome
                transform is used.
                If None and using an inferred noise likelihood, the noise will be the
                inferred noise level. If using a fixed noise likelihood, the mean across
                the observation noise in the training data is used as observation noise.
            evaluation_mask: A `n' x m(d)-dim tensor of booleans indicating which
                outputs should be fantasized for a given design. This uses the same
                evaluation mask for all batches.  evaluation_mask[0,j] indicating for candidate 0_th, input_mask for
                j_th output.
            kwargs: Will be passed to `model.condition_on_observations`

        Returns:
            The constructed fantasy model.
        """
        # if the inputs are empty, expand the inputs
        if X.shape[-2] == 0:
            output_shape = (
                sampler.sample_shape
                + X.shape[:-2]
                + torch.Size([0, self.num_outputs])
            )
            Y = torch.empty(output_shape, dtype=X.dtype, device=X.device)
            return self.condition_on_observations(X=X,Y=Y,**kwargs)

        propagate_grads = kwargs.pop("propagate_grads", False)
        with fantasize_flag():
            with settings.propagate_grads(propagate_grads):
                post_X = self.posterior(
                    X,
                    observation_noise=True
                    if observation_noise is None
                    else observation_noise,
                )
            Y_fantasized = sampler(post_X) # num_fantasies x batch_shape x n' x m
            return self.condition_on_observations(X, Y_fantasized, **kwargs)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        r"""Condition the model on new observations.
        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n' x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, it is assumed that the missing batch dimensions are
                the same for all `Y`.
        Returns:
            A `Model` object of the same type, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).
        """
        fantasy_models:List[ SingleTaskGP|None] = [None for k in range(self.n_nodes)]

        for k in self.root_nodes:
            if self.active_indices is not None:
                X_node_k = X[..., self.active_indices[k]]
            else:
                X_node_k = X
            Y_node_k = Y[..., [k]]
            fantasy_models[k] = self.node_GPs[k].condition_on_observations(X_node_k, Y_node_k,
                                                                           noise=torch.ones(Y_node_k.shape[1:]) * 1e-6)
        for k in range(self.n_nodes):
            if fantasy_models[k] is None:
                aux = Y[..., self.dag.get_parent_nodes(k)].clone()
                for j in range(len(self.dag.get_parent_nodes(k))):
                    aux[..., j] = (aux[..., j] - self.Z_lr[k][j])/(self.Z_up[k][j] - self.Z_lr[k][j])

                aux_shape=(aux.shape[0],)+ (1,) * X.ndim
                X_node_k = torch.cat([X[None,..., self.active_indices[k]].repeat(*aux_shape),aux], -1)
                Y_node_k = Y[..., [k]]
                fantasy_models[k] = self.node_GPs[k].condition_on_observations(X_node_k, Y_node_k,
                                                                               noise=torch.ones(Y_node_k.shape[1:]) * 1e-6)

        return GaussianProcessNetwork(dag=self.dag, train_X=X, train_Y=Y, active_indices=self.active_indices, node_GPs=fantasy_models, Z_lr=self.Z_lr, Z_up=self.Z_up)

#For posterior
class MultivariateNormalNetwork(Posterior):
    def __init__(self, node_GPs, dag, X, indices_X=None, Z_lr=None, Z_up=None):
        self.node_GPs = node_GPs
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.X = X
        self.active_indices = indices_X
        self.Z_lr = Z_lr
        self.Z_up = Z_up

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return torch.double

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        shape = list(self.X.shape)
        shape[-1] = self.n_nodes
        shape = torch.Size(shape)
        return shape

    @property
    def batch_range(self):
        r"""The t-batch range.

        This is used in samplers to identify the t-batch component of the
        `base_sample_shape`. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        return (0, -2)

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The shape of a base sample used for constructing posterior samples.
                         :(sample_size,num_restart, num_candidate, output_dim+inputdim),
         real base sample: (sample_size, 1,num_candidate, output_dim+inputdim)
        This function may be overwritten by subclasses in case `base_sample_shape`
        and `event_shape` do not agree (e.g. if the posterior is a Multivariate
        Gaussian that is not full rank).
        """
        shape = list(self.X.shape)
        shape[-1] = self.n_nodes
        shape = torch.Size(shape)
        return shape

    def rsample_from_base_samples(self, sample_shape=torch.Size(), base_samples=None):
        #base_samples=base_samples[:,[0],:,:]
        nodes_samples = torch.empty(sample_shape + self.event_shape)
        nodes_samples = nodes_samples.double()
        nodes_samples_available = [False for k in range(self.n_nodes)]
        for k in self.root_nodes:
            if self.active_indices is not None:
                X_node_k = self.X[..., self.active_indices[k]]
            else:
                X_node_k = self.X
            mvn_node_k = self.node_GPs[k].posterior(X_node_k)
            if base_samples is not None:# then use reparametrization trick
                nodes_samples[..., k] = mvn_node_k.rsample(sample_shape, base_samples=base_samples[..., [k]])[..., 0]
            else:                      # sample from mean and coviraince function of the GP
                nodes_samples[..., k] = mvn_node_k.rsample(sample_shape)[..., 0]
            nodes_samples_available[k] = True

        while not all(nodes_samples_available):
            for k in range(self.n_nodes):
                parent_nodes = self.dag.get_parent_nodes(k)
                if not nodes_samples_available[k] and all([nodes_samples_available[j] for j in parent_nodes]):
                    parent_samples_norm = nodes_samples[..., parent_nodes].clone()
                    for j in range(len(parent_nodes)):
                        parent_samples_norm[..., j] =  (parent_samples_norm[..., j] - self.Z_lr[k][j])/(self.Z_up[k][j] - self.Z_lr[k][j])
                    X_node_k = self.X[..., self.active_indices[k]]
                    aux_shape = [sample_shape[0]] + [1] * X_node_k.ndim
                    X_node_k = X_node_k.unsqueeze(0).repeat(*aux_shape) # repeat to consistent with sample size 128
                    mvn_node_k = self.node_GPs[k].posterior(torch.cat([X_node_k, parent_samples_norm.clip(0.,1.)], -1))
                    if base_samples is not None:
                        nodes_samples[...,[k]] = mvn_node_k.mean +torch.sqrt(mvn_node_k.variance) * base_samples[..., [k]]
                    else:
                        nodes_samples[..., [k]] = mvn_node_k.rsample()[0,...,0]
                    nodes_samples_available[k] = True
        return nodes_samples

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        nodes_samples=self.rsample_from_base_samples(sample_shape,base_samples)
        return nodes_samples

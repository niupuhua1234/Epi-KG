from __future__ import annotations
import torch
from typing import Any,Optional,List
from botorch.models.model import Model
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


class GaussianProcessSeq(Model):
    def __init__(self,n_node,T, train_X, train_Y,train_Yvar=None,node_GPs:List[SingleTaskGP]=None,evaluation_mask=None) -> None:
        super(Model, self).__init__()  ####
        self.n_nodes=n_node
        self.T=T
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar =torch.full_like(train_Y, 1e-7)
        self.evaluation_mask:Optional[None|Tensor]=evaluation_mask

        if node_GPs is not None:
            self.node_GPs:List[ SingleTaskGP]  = node_GPs
        else:
            self.node_GPs:List[ SingleTaskGP|None]  = [None for _ in range(self.n_nodes)]
            self.node_mlls:List[ExactMarginalLogLikelihood|None] = [None for _ in range(self.n_nodes)]
            for k in range(self.n_nodes):
                self.node_GPs[k] =FixedNoiseGP(self.train_X, self.train_Y[...,k], self.train_Yvar[...,k],
                                               outcome_transform=Standardize(m=train_Y[...,k].shape[-1]))
                # fit the GP for each node
                self.node_mlls[k] = ExactMarginalLogLikelihood(self.node_GPs[k].likelihood, self.node_GPs[k])
                fit_gpytorch_model(self.node_mlls[k])
    @property
    def num_outputs(self) -> int:
        return self.n_nodes*self.train_Y[0].shape[-1]

    def posterior(self, X: Tensor,output_indices:int = 0, observation_noise=False, **kwargs) -> MultivariateNormalSeq:
        return MultivariateNormalSeq(self.node_GPs, X,output_indices=output_indices,n_nodes=self.n_nodes,T=self.T)

    def forward(self, X: Tensor,output_indices:int = 0) -> MultivariateNormalSeq:
        return MultivariateNormalSeq(self.node_GPs, X,output_indices=output_indices,n_nodes=self.n_nodes,T=self.T)

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
            Y_fantasized = sampler(post_X)
            Y_fantasized =Y_fantasized.reshape(Y_fantasized.shape[:-1]+(self.T,self.n_nodes))# num_fantasies x batch_shape x n' x m
            return self.condition_on_observations(X, Y_fantasized,self.evaluation_mask, **kwargs)

    def condition_on_observations(self, X: Tensor, Y: Tensor,evaluation_mask = None, **kwargs: Any) -> Model:
        fantasy_models:List[ SingleTaskGP|None] = [None for _ in range(self.n_nodes)]
        for k in range(self.n_nodes):
            X_node_k = X[..., evaluation_mask[:,k],:]  if evaluation_mask is not None else X
            Y_node_k = Y[...,k][...,evaluation_mask[:,k],:] if evaluation_mask is not None else Y[...,k]
            fantasy_models[k] = self.node_GPs[k].condition_on_observations(X_node_k,Y_node_k,
                                                                           noise=torch.ones(Y_node_k.shape[1:]) * 1e-6)
        return GaussianProcessSeq(n_node=self.n_nodes,T=self.T,train_X=X, train_Y=Y,node_GPs=fantasy_models)


class MultivariateNormalSeq(Posterior):
    def __init__(self, node_GPs, X, output_indices,n_nodes=4, T=30):
        self.node_GPs = node_GPs
        self.X = X
        self.n_nodes = n_nodes
        self.T = T
        self.output_indices = output_indices

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return torch.double

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The shape of a base sample used for constructing posterior samples.
                         :(sample_size,num_restart, num_candidate, output_dim+inputdim),
         real_us base sample: (sample_size, 1,num_candidate, output_dim+inputdim)
        This function may be overwritten by subclasses in case `base_sample_shape`
        and `event_shape` do not agree (e.g. if the posterior is a Multivariate
        Gaussian that is not full rank).
        """
        shape = self.X.shape[:-1] + (self.T, self.n_nodes)
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
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        shape = self.X.shape[:-1] + (self.T, self.n_nodes)
        shape = torch.Size(shape)
        return shape

    def rsample_from_base_samples(self, sample_shape=torch.Size(), base_samples=None):
        nodes_samples = torch.empty(sample_shape + self.event_shape)
        nodes_samples = nodes_samples.double()
        for k in range(self.n_nodes):
            mvn_at_node_k = self.node_GPs[k].posterior(self.X)
            if base_samples is not None:
                nodes_samples[..., k] = mvn_at_node_k.rsample(sample_shape, base_samples=base_samples[..., k])
            else:
                nodes_samples[..., k] = mvn_at_node_k.rsample(sample_shape)
        return nodes_samples.reshape(nodes_samples.shape[:-2] + (-1,))

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        nodes_samples=self.rsample_from_base_samples(sample_shape,base_samples)
        return nodes_samples

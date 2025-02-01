import torch
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, qKnowledgeGradient,GenericMCObjective
from botorch.acquisition import PosteriorMean as GPPosteriorMean
#from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor
from typing import List

from bofn.acquisition_function_optimization.optimize_acqf import optimize_acqf_and_get_suggested_point,optimize_decoupled_KG_and_get_suggested_point
from bofn.utils.dag import DAG
from bofn.utils.fit_gp_model import fit_gp_model
from bofn.models.gp_network import GaussianProcessNetwork
from bofn.models.gp_seq import GaussianProcessSeq
from bofn.utils.posterior_mean import PosteriorMean


def get_new_suggested_point(
    algo: str,
    X: Tensor,
    Y: Tensor,
    bounds:Tensor,
    obj: Tensor,
    obj_transform: GenericMCObjective,
    dag: DAG,
    active_indices: List[int],
) -> Tensor:
    input_dim = X.shape[-1]

    if algo == "Random":
        return torch.rand([1, input_dim])

    elif algo == "EICF":
        model = fit_gp_model(X=X, Y=Y)
        #qmc_sampler = SobolQMCNormalSampler(num_samples=128) # provided base samples for reparametrization trick
        qmc_sampler = SobolQMCNormalSampler(torch.Size([128]))
        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=obj.max().item(),
            sampler=qmc_sampler,
            objective=obj_transform,
        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=obj_transform,
        )
    elif algo == "EI":
        model = fit_gp_model(X=X, Y=obj)
        acquisition_function = ExpectedImprovement(
            model=model, best_f=obj.max().item())
        posterior_mean_function = GPPosteriorMean(model=model)

    elif algo == "KG":
        model = fit_gp_model(X=X, Y=obj)
        #qmc_sampler = SobolQMCNormalSampler(num_samples=128) # provided base samples for reparametrization trick
        qmc_sampler = SobolQMCNormalSampler(torch.Size([128]))
        acquisition_function = qKnowledgeGradient(
            model=model,
            inner_sampler=qmc_sampler,
            num_fantasies=8)
        posterior_mean_function = GPPosteriorMean(model=model)

    elif algo == "KGCF":
        model = fit_gp_model(X=X, Y=Y)
        #qmc_sampler = SobolQMCNormalSampler(num_samples=128) # provided base samples for reparametrization trick
        qmc_sampler = SobolQMCNormalSampler(torch.Size([128]))
        acquisition_function    = qKnowledgeGradient(
            model=model,
            inner_sampler=qmc_sampler,
            objective=obj_transform,
            num_fantasies=8)
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=obj_transform,
        )

    elif algo == "KGFN":
        # Model
        model = GaussianProcessNetwork(train_X=X,
                                       train_Y=Y,
                                       dag=dag,
                                       active_indices=active_indices)
        # Sampler
        #qmc_sampler = SobolQMCNormalSampler(num_samples=128)
        qmc_sampler = SobolQMCNormalSampler(torch.Size([128]))
        # Acquisition function
        acquisition_function    = qKnowledgeGradient(
            model=model,
            inner_sampler=qmc_sampler,
            objective=obj_transform,
            num_fantasies=8)
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=obj_transform,
        )

    elif algo=='DGCF':
        n_nodes=len(dag.root_nodes)
        T=dag.n_nodes//n_nodes
        Y=Y.reshape(-1,T, n_nodes)
        model = GaussianProcessSeq(n_node= n_nodes,T=T,train_X=X,train_Y=Y)
        #qmc_sampler = SobolQMCNormalSampler(num_samples=128) # provided base samples for reparametrization trick
        qmc_sampler = SobolQMCNormalSampler(torch.Size([128]))
        acquisition_setup    = {'model':model,
                                'inner_sampler':qmc_sampler,
                                'objective':obj_transform,
                                'num_fantasies':8}
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=obj_transform,
        )
    else:
        raise ValueError('No BO methods are selected!!!')

    if algo!='DGCF':
        new_x = optimize_acqf_and_get_suggested_point(
            acq_func=acquisition_function,
            bounds=bounds,
            posterior_mean=posterior_mean_function, # for baseline
        )
    else:
        new_x = optimize_decoupled_KG_and_get_suggested_point(
            **acquisition_setup,
            bounds=bounds,
            posterior_mean=posterior_mean_function, # for baseline
        )
    return new_x




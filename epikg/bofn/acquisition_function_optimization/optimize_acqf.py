import time

import torch
from torch import Tensor
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.optim import optimize_acqf
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)

def optimize_acqf_and_get_suggested_point(
    acq_func,
    bounds,
    batch_size=1,
    posterior_mean=None,
    ) -> Tensor:
    """Optimizes the acquisition function, and returns a new candidate."""
    input_dim   =bounds.shape[1]
    num_restarts=5#*bounds.shape[1]  # The number of starting points for multistart acquisition function optimization.
    raw_samples= 100#*bounds.shape[1]  #The number of samples for initialization.
                               # This is required if `batch_initial_conditions` is not specified.

    ic_gen= gen_one_shot_kg_initial_conditions  if isinstance(acq_func, qKnowledgeGradient) \
        else gen_batch_initial_conditions
    #  gen_one_shot_kg_initial_conditions 0.1 for random initial x0, 0.8 for x0 that maximize the current posterior

    batch_initial_conditions = ic_gen(
        acq_function=acq_func, ####
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit":num_restarts}, # if batch_dim of X or Y is not none, it should be set to equal to batch_dim.
    )

    if posterior_mean is not None:
        baseline_candidate, current_vals = optimize_acqf(
            acq_function=posterior_mean,  ######
            bounds=bounds,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5},
        )
        if isinstance(acq_func, qKnowledgeGradient):#Knowledge Gradient
            acq_func.current_value=current_vals
            augmented_q_batch_size = acq_func.get_augmented_q_batch_size(batch_size)# q_number of fantasy_number
            baseline_candidate = baseline_candidate.detach().repeat(1, augmented_q_batch_size, 1)
        else:
            baseline_candidate = baseline_candidate.detach().view(torch.Size([1, batch_size, input_dim]))
        # combine the initioal conditions and baseline——candidate
        batch_initial_conditions = torch.cat([batch_initial_conditions, baseline_candidate], 0)
        num_restarts += 1
    t0=time.time()
    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={"batch_limit": 6},
        #options={'disp': True, 'iprint': 101},
    )
    t1=time.time()
    print(t1-t0)

    if posterior_mean is not None:
        baseline_acq_value = acq_func.forward(baseline_candidate)[0].detach()
        print('Baseline function value', baseline_acq_value.item())
        print('Acquisition function value', acq_value.item())
        if baseline_acq_value >= acq_value:
            print('Baseline candidate was best found.')
            candidate= baseline_candidate[:,0,:] if isinstance(acq_func, qKnowledgeGradient) \
                else baseline_candidate

    new_x = candidate.detach().view([batch_size, input_dim])
    return new_x




def optimize_decoupled_KG_and_get_suggested_point(
    model,
    inner_sampler,
    objective,
    num_fantasies,
    bounds,
    batch_size=1,
    posterior_mean=None,
    ) -> Tensor:
    """Optimizes the acquisition function, and returns a new candidate."""
    input_dim   =bounds.shape[1]
    num_restarts=5#*bounds.shape[1]  # The number of starting points for multistart acquisition function optimization.
    raw_samples= 100#*bounds.shape[1]  #The number of samples for initialization.
                               # This is required if `batch_initial_conditions` is not specified.

    baseline_candidate, current_vals = optimize_acqf(
        acq_function=posterior_mean,  ######
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5},
    )
    num_restarts += 1

    acq_vals = []
    acq_candidates = []
    for objective_idx in range(5):
        if objective_idx!=4:
            model.evaluation_mask= torch.zeros(1, 4, dtype=torch.bool)
            model.evaluation_mask[:, objective_idx] = 1
        else:
            model.evaluation_mask=None
        acq_func = qKnowledgeGradient(
            model=model,
            inner_sampler=inner_sampler,
            num_fantasies=num_fantasies,
            objective=objective,
            current_value=current_vals,
        )
        t0=time.time()
        ic_gen= gen_one_shot_kg_initial_conditions
        batch_initial_conditions = ic_gen(
            acq_function=acq_func, ####
            bounds=bounds,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit":num_restarts}, # if batch_dim of X or Y is not none, it should be set to equal to batch_dim.
        )

        augmented_q_batch_size = acq_func.get_augmented_q_batch_size(batch_size)  # q_number of fantasy_number
        # combine the initioal conditions and baseline——candidate
        baseinit_candidate=baseline_candidate.detach().repeat(1, augmented_q_batch_size, 1)
        batch_initial_conditions = torch.cat([batch_initial_conditions, baseinit_candidate], 0)

        candidate, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            batch_initial_conditions=batch_initial_conditions,
            options={"batch_limit": 6},
            #options={'disp': True, 'iprint': 101},
        )
        acq_vals.append(acq_value)
        acq_candidates.append(candidate)
        t1=time.time()
        print(t1-t0)


    best_index =torch.stack(acq_vals).argmax().item()
    candidate = acq_candidates[best_index]


    baseline_acq_value = acq_func.forward(baseinit_candidate)[0].detach()
    print('Baseline function value', baseline_acq_value.item())
    print('Acquisition function value', acq_vals[best_index].item())
    if baseline_acq_value >= acq_vals[best_index]:
        print('Baseline candidate was best found.')
        candidate=  baseinit_candidate[:,0,:]

    new_x = candidate.detach().view([batch_size, input_dim])
    return new_x





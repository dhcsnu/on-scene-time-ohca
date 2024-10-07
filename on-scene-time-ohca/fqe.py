#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import d3rlpy
from d3rlpy.metrics import TDErrorEvaluator, InitialStateValueEstimationEvaluator
from d3rlpy.ope.fqe import DiscreteFQE
import numpy as np
import random
import torch

def fit_fqe(test_dataset, policy, fqe_config, epochs, steps_epoch, save_path=None, show_progress=True, seed=0):
    
    d3rlpy.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fqe = DiscreteFQE(algo=policy, config=fqe_config, device="cuda:0")
    fqe.create_impl(policy.observation_shape, policy.action_size)
    
    print("\nFQE fitting...\n")
    
    fqe.fit(test_dataset, n_steps=steps_epoch*epochs, n_steps_per_epoch=steps_epoch, show_progress=show_progress, 
         evaluators={
               'td_error': TDErrorEvaluator(test_dataset.episodes),
               'initial_state_value': InitialStateValueEstimationEvaluator(test_dataset.episodes)})
    
    if save_path is not None:
        fqe.save_model(save_path)
    
    return fqe


#!/usr/bin/env python
# coding: utf-8

# In[5]:


import d3rlpy
from d3rlpy.algos.qlearning.base import QLearningAlgoBase
from d3rlpy.algos.qlearning.cql import DiscreteCQLConfig
from d3rlpy.algos.qlearning.torch.dqn_impl import DQNModules
from d3rlpy.algos.qlearning.torch import DiscreteCQLImpl
from d3rlpy.base import DeviceArg, LearnableConfig, register_learnable
from d3rlpy.constants import ALGO_NOT_GIVEN_ERROR, ActionSpace
from d3rlpy.metrics import TDErrorEvaluator, DiscreteActionMatchEvaluator, InitialStateValueEstimationEvaluator
from d3rlpy.models import VectorEncoderFactory
from d3rlpy.models.builders import create_discrete_q_function
from d3rlpy.types import NDArray, Observation, Shape
import numpy as np
import random
import torch


class OptimalDiscreteCQLImpl(DiscreteCQLImpl):    
    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DQNModules,
        q_func_forwarder,
        targ_q_func_forwarder,
        target_update_interval: int,
        gamma: float,
        alpha: float,
        device: str,
        t_min,
        t_max
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            target_update_interval=target_update_interval,
            gamma=gamma,
            alpha=alpha,
            device=device
        )
        
        self.t_min = t_min
        self.t_max = t_max
        
    def inner_predict_best_action(self, x) -> torch.Tensor:
        actions = self._q_func_forwarder.compute_expected_q(x).argmax(dim=1).clone().to(torch.int64)
        rosc = x[:,-1].to(torch.int64)
        tmax_bool = (x[:,-2] >= self.t_max).to(torch.int64)
        tmin_bool = (x[:,-2] >= self.t_min).to(torch.int64)
        max_action = (actions | rosc | tmax_bool) & tmin_bool
        return max_action

class OptimalDiscreteCQL(QLearningAlgoBase[OptimalDiscreteCQLImpl, DiscreteCQLConfig]):
    def __init__(self, *args, t_min, t_max, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_min = t_min
        self.t_max = t_max
    
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        q_funcs, q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        targ_q_funcs, targ_q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )

        optim = self._config.optim_factory.create(
            q_funcs.named_modules(), lr=self._config.learning_rate
        )

        modules = DQNModules(
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            optim=optim,
        )

        self._impl = OptimalDiscreteCQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            target_update_interval=self._config.target_update_interval,
            gamma=self._config.gamma,
            alpha=self._config.alpha,
            device=self._device,
            t_min=self.t_min,
            t_max=self.t_max
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE

class OptimalDiscreteCQLConfig(DiscreteCQLConfig):
    """
    t_min: The normalized time before which intra-arrest transport cannot be started.
    t_max: The normalized time after which all patients are transported intra-arrest, even without ROSC.
    """
    def __init__(self, *args, t_min, t_max, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_min = t_min
        self.t_max = t_max
    
    def create(self, device: DeviceArg = False):
        return OptimalDiscreteCQL(self, device, t_min=self.t_min, t_max=self.t_max)
    

def train_policy(train_buffer, val_dataset, policy, epochs, steps_epoch,
                 target_update_interval=None, show_progress=True, seed=0):
    
    d3rlpy.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if target_update_interval is None:
        target_update_interval = int(steps_epoch/2)

    print("\nCQL fitting...\n")
    
    policy.fit(train_buffer, n_steps=steps_epoch*epochs, n_steps_per_epoch=steps_epoch, show_progress=show_progress, 
         evaluators={
               'td_error': TDErrorEvaluator(val_dataset.episodes),
               'action_match': DiscreteActionMatchEvaluator(val_dataset.episodes),
               'initial_state_value': InitialStateValueEstimationEvaluator(val_dataset.episodes),
           })
    
    return policy


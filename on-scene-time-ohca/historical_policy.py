#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import defaultdict
from d3rlpy.algos.qlearning.base import QLearningAlgoBase
from d3rlpy.algos.qlearning.cql import DiscreteCQLConfig
from d3rlpy.algos.qlearning.torch import DiscreteCQLImpl
from d3rlpy.algos.qlearning.torch.dqn_impl import DQNModules
from d3rlpy.base import DeviceArg, LearnableConfig, register_learnable
from d3rlpy.constants import ALGO_NOT_GIVEN_ERROR, ActionSpace
from d3rlpy.models.builders import create_discrete_q_function
from d3rlpy.types import NDArray, Observation, Shape
import numpy as np
import random
import torch


def make_hash_table(dataset):
    hash_table = defaultdict(list)
    
    for episode in dataset.episodes:
        for index in range(len(episode.observations)):
            observation_tuple = tuple(np.int16(episode.observations[index]*10000).tolist())
            hash_table[observation_tuple].append(episode.actions[index])
            
    return hash_table


class HistoricalDiscreteCQLImpl(DiscreteCQLImpl):   
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
        hash_table
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
        
        self.hash_table = hash_table
    
    def inner_predict_best_action(self, x):
        actions = self.historical_actions(x)
        return torch.tensor(actions, dtype=torch.int64).cuda()

    def historical_actions(self, xs):
        xs = xs.cpu()

        actions = []
        for x in xs:
            x_tuple = tuple(np.int16(x*10000))
            y_list = self.hash_table.get(x_tuple, [])
            if not y_list:
                actions.append([0])
            else:
                index = np.random.randint(len(y_list))
                actions.append(y_list[index])
        return np.int64(actions).flatten()

    
class HistoricalDiscreteCQL(QLearningAlgoBase[HistoricalDiscreteCQLImpl, DiscreteCQLConfig]):
    def __init__(self, *args, hash_table, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash_table = hash_table
        
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

        self._impl = HistoricalDiscreteCQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            target_update_interval=self._config.target_update_interval,
            gamma=self._config.gamma,
            alpha=self._config.alpha,
            device=self._device,
            hash_table=self.hash_table
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE

       
class HistoricalDiscreteCQLConfig(DiscreteCQLConfig):
    def __init__(self, *args, hash_table, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash_table = hash_table
        
    def create(self, device: DeviceArg = False):
        return HistoricalDiscreteCQL(self, device, hash_table=self.hash_table)


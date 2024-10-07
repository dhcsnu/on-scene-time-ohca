#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import d3rlpy
from d3rlpy.dataset import MDPDataset, FIFOBuffer
from IPython.display import clear_output
import numpy as np
import pandas as pd

def build_MDPDataset(data, hazard_function, famd, rosc_svv, state_columns, cycle_mean, cycle_std,
                       max_steps=16, reward_type='svv', hazard_reward_ratio=1):

    data_processed = famd.transform(data[state_columns])
    
    def norm_time(x):
        return (x - cycle_mean) / cycle_std
        
    # Segment episodes by timeframe
    # State : [demographics, time, ROSC state]

    observations, actions, rewards, terminals = [], [], [], []

    for episode in range(len(data)):
        state_data = data_processed[episode,:]
        srti = data['crosc_time'].iloc[episode]   # time (cycle) from scene arrival to on-scene ROSC
        srosc = data['s_rosc'].iloc[episode]      # on-scene rosc
        sti = data['csti'].iloc[episode]          # time (cycle) from scene arrival to intra-arrest transport 
        
        # Rewards
        hazard_reward = hazard_function.iloc[episode]                      # Intermediate rewards
        final_reward = 1 if data[reward_type].iloc[episode] == 1 else 0   # Final rewards
        reward_sum = 0                                                     # Initialize reward sum
        
        if srosc == 1:   
            for step in range(max_steps):
                if step < srti:
                    state = np.append(state_data, [norm_time(step), 0])
                    observations.append(state)
                    actions.append(0)
                    reward = hazard_reward[step]*rosc_svv[step] 
                    rewards.append(reward * hazard_reward_ratio)
                    reward_sum += reward * hazard_reward_ratio
                    terminals.append(0) 
                elif step == srti:
                    state = np.append(state_data, [norm_time(step), 1])
                    observations.append(state)
                    actions.append(1)
                    rewards.append(final_reward - reward_sum)
                    terminals.append(1)
                
        elif srosc == 0:   
            for step in range(max_steps):
                if step < sti:
                    state = np.append(state_data, [norm_time(step), 0])
                    observations.append(state)
                    actions.append(0)
                    reward = hazard_reward[step]*rosc_svv[step] 
                    rewards.append(reward * hazard_reward_ratio)
                    reward_sum += reward * hazard_reward_ratio
                    terminals.append(0)                             
                elif step == sti:
                    state = np.append(state_data, [norm_time(step), 0])
                    observations.append(state)
                    actions.append(1)
                    rewards.append(final_reward - reward_sum)
                    terminals.append(1)
  
    clear_output(wait=False)
    
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    
    original_dataset = MDPDataset(observations, actions, rewards, terminals)
   
    return original_dataset

def build_buffer(dataset):
    buffer = d3rlpy.dataset.FIFOBuffer(limit=1000000)
    transition_picker = d3rlpy.dataset.BasicTransitionPicker()
    trajectory_slicer = d3rlpy.dataset.BasicTrajectorySlicer()
    writer_preprocessor = d3rlpy.dataset.BasicWriterPreprocess()
    
    buffer_dataset = d3rlpy.dataset.ReplayBuffer(
        buffer=buffer,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
        writer_preprocessor=writer_preprocessor,
        episodes=dataset.episodes)
    
    return buffer_dataset


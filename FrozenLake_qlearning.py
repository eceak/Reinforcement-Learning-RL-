#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:45:11 2021

@author: ece
"""

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0").env


# %%
print("State space:",env.observation_space)#there are 16 discrete state
print("Action space",env.action_space)#there are 4 discerete action space

# %%
from gym.envs.registration import register
register(
    id="FrozenLakeNotSlippery-v0",
    entry_point="gym.env.toy_text:FrozenLakeEnv",
    kwargs={"map_name" : '4*4' , 'is_slippery:': False},
    max_episode_steps=100,
    reward_threshold=0.78,
    )

#Q table
q_table = np.zeros([env.observation_space.n,env.action_space.n])#16 discrete states and 4 discerete actions

#Hyperparameter
alpha = 0.8
gamma = 0.95
epsilon = 0.1

#plotting Metrix
reward_list = []


episode_number = 10000
for i in range(1,episode_number):
    
    #initialize environment
    state = env.reset()
    
    reward_count = 0
    
    
    
    while True :
        
        #exploit vs explore to find action
        # %10 explore,%90 exploit
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        #action process and take reward/observation
        next_state, reward, done, _ = env.step(action)
        
        #Q learning function
        old_value = q_table[state,action] #old_value
        next_max = np.max(q_table[next_state]) #next_max
        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        
        #QTable update
        q_table[state,action] = next_value
        
        
        #update state
        state = next_state
        
        reward_count += reward
       
    
        
        if done:
            break
        
    if i%10 == 0:
      
      reward_list.append(reward_count)
      print("Episode: {},reward: {}".format(i,reward_count,))
        
# %% visualize
plt.plot(reward_list)



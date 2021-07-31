
"""
Q learning at RL -taxi implementation-

@author: ece
"""

import gym

env = gym.make("Taxi-v3").env

env.render() #show

'''
blue = passenger
purple = destination
yellow/red = empty taxi 
green = full taxi
RGBY = location for destination and passenger
'''
env.reset() #reset env and return random initial state

# %%
print("State space:",env.observation_space)#there are 500 discrete states
print("Action space",env.action_space)#there are 6 discerete space

#taxi row,taxi column,passenger index,destination
state = env.encode(3,1,2,3)
print("State number:",state)

env.s = state
env.render()

# %%
#probability,next_state,reward,done
env.P[331]

#%%
#for 1 episode 

env.reset()
time_step = 0 
total_reward = 0
list_visualize = []

while True:
    
    time_step += 1
    
    #choose action
    action = env.action_space.sample()
    
    #perform action and get reward
    state, reward, done, _ = env.step(action) #state=next_state
    
    #total reward
    total_reward += reward  
    
    #visualize
    list_visualize.append({"frame": env,
                           "state": state, "action" : action, "reward": reward, 
                           "total reward": total_reward})
    #env.render()
    
    if done:
        break
    
    # %%
import time #çok hızlı gösterilmesi engellenmek için bu kütüphane tanımlandı

for i, frame in enumerate(list_visualize):
    print(frame["frame"].render())
    print("Timestep:", i + 1)
    print("State:", frame["state"])
    print("action:", frame["action"])
    print("reward:", frame["reward"])
    print("total reward:", frame["total reward"])
    
    time.sleep(2)
    
    
    
    
    


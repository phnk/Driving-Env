import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


# Load data into dict called data
fname = ['trpo', 'ppo', 'reinforce']
base = '_v0_rew'
for ind in range(len(fname)):
    fname[ind] = f'{fname[ind]}{base}'
data = {}
for name in fname: 
    with open(name, 'rb') as f: 
        data.update(pickle.load(f))
for k, v in data.items(): 
    print(f'{k} episodes: {len(v)}')

x_axis = [x for x in range(800)]
plt.figure(figsize=(10, 6))

# reinforce
smooth = gaussian_filter1d(data['reinforce'], sigma=20)
plt.plot(x_axis, smooth, label='REINFORCE')
plt.scatter(x_axis, data['reinforce'], s=2)

# ppo
smooth = gaussian_filter1d(data['ppo'], sigma=20)
plt.scatter(x_axis, smooth, label='PPO', s=2, marker='D')
plt.scatter(x_axis, data['ppo'], s=2)

# trpo
smooth = gaussian_filter1d(data['trpo'], sigma=20)
plt.scatter(x_axis, smooth, label='TRPO')
plt.scatter(x_axis, data['trpo'], s=2)

plt.title('Driving-v0 Episode Reward over Training')
plt.xlabel('Episodes')
plt.ylabel('Environment Reward')
plt.legend(loc='upper left')
plt.show()



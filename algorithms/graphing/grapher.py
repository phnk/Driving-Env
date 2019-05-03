import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


# Load data into dict called data
fname = ['trpo', 'ppo', 'reinforce']
base = '_v1_rew'
for ind in range(len(fname)):
    fname[ind] = f'{fname[ind]}{base}'
data = {}
for name in fname: 
    with open(name, 'rb') as f: 
        data.update(pickle.load(f))
for k, v in data.items(): 
    print(f'{k} episodes: {len(v["rewards"])}', end='\t')
    print(f'{k} 0 reward: {len([0 for val in v["rewards"] if abs(val) < 0.2])}')
for info in data.values():
    lens = info['lengths']
    for ind in range(1, len(lens)):
        lens[ind] += lens[ind - 1] 

plt.figure(figsize=(10, 6))

# reinforce
smooth = gaussian_filter1d(data['reinforce']['rewards'], sigma=20)
plt.scatter(data['reinforce']['lengths'], smooth, color='g', s=3)
plt.scatter(data['reinforce']['lengths'], data['reinforce']['rewards'], s=2,
color='g', label='REINFORCE')

# ppo
smooth = gaussian_filter1d(data['ppo']['rewards'], sigma=20)
plt.scatter(data['ppo']['lengths'], smooth,  color='r', s=3)
plt.scatter(data['ppo']['lengths'], data['ppo']['rewards'], s=2, color='r', label='PPO')

# trpo
smooth = gaussian_filter1d(data['trpo']['rewards'], sigma=20)
plt.scatter(data['trpo']['lengths'], smooth, color='c', s=3)
plt.scatter(data['trpo']['lengths'], data['trpo']['rewards'], label='TRPO', s=2,
color='c')

plt.title('Driving-v1 Reward over Training')
plt.xlabel('Environment Steps')
plt.ylabel('Environment Reward')
plt.xticks(np.arange(0, 960001, 120000))
plt.legend(loc='upper left', markerscale=4)
plt.show()

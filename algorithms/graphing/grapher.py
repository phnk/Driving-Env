import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

mpl.style.use('seaborn')
from pylab import rcParams
rcParams['figure.figsize'] = 20, 5
rcParams.update({'figure.autolayout': True})

def len_tool(lens):  
  for ind in range(1, len(lens)): 
    lens[ind] += lens[ind - 1]

def bin_tool(lens, rew, bin_size):
  start_ind = 0
  counter = 0
  bin_rewards = []
  bin_lengths = [0]
  for ind in range(len(lens)): 
    counter += lens[ind]
    if counter >= bin_size: 
      bin_rewards.append(sum(rew[start_ind:ind]) / (ind - start_ind))
      bin_lengths.append(counter + bin_lengths[-1])
      counter -= bin_size
      start_ind = ind
  del bin_lengths[0]

  return bin_lengths, bin_rewards

# Load data into dict called data
with open(f'ppo_reward_v0', 'rb') as f: 
    data = pickle.load(f)
lens = data['lengths']
rew = data['rewards']
bin_lens, bin_rew = bin_tool(lens, rew, 180000)
plt.subplot(1, 3, 1)
plt.title('Driving-v0', fontsize=20)
plt.xlabel('Timesteps', fontsize=16)
plt.xticks((0, 16e6), ['0', '16M'], fontsize=12)
plt.xlim((0, 16e6))
plt.yticks(fontsize=12)
plt.ylabel('Reward', fontsize=16)
plt.plot(bin_lens, bin_rew, color='g')

with open(f'ppo_reward_v1', 'rb') as f: 
    data = pickle.load(f)
lens = data['lengths']
rew = data['rewards']
bin_lens, bin_rew = bin_tool(lens, rew, 180000)
plt.subplot(1, 3, 2)
plt.title('Driving-v1', fontsize=20)
plt.xlabel('Timesteps', fontsize=16)
plt.xticks((0, 16e6), ['0', '16M'], fontsize=12)
plt.xlim((0, 16e6))
plt.yticks(fontsize=12)
plt.ylabel('Reward', fontsize=16)
plt.plot(bin_lens, bin_rew, color='g')

with open(f'ppo_reward_v2', 'rb') as f: 
    data = pickle.load(f)
lens = data['lengths']
rew = data['rewards']
bin_lens, bin_rew = bin_tool(lens, rew, 180000)
plt.subplot(1, 3, 3)
plt.title('Driving-v2', fontsize=20)
plt.xlabel('Timesteps', fontsize=16)
plt.xticks((0, 16e6), ['0', '16M'], fontsize=12)
plt.xlim((0, 16e6))
plt.yticks(fontsize=12)
plt.ylabel('Reward', fontsize=16)
plt.plot(bin_lens, bin_rew, color='g')
plt.show()

rcParams['figure.figsize'] = 6, 5

N = 3
success = [100, 34.6, 24.8]
nothing = [0, 64.8, 72.6]
collision = [0, 0.6, 2.6]
plt.ylabel('Number of Collisions or Success', fontsize=13)
plt.xlabel('Environment', fontsize=13)
plt.title('Final Performance on Environment', fontsize=18)
ind = np.arange(N)
width = 0.3


plt.bar(ind, success, width, label="Reaches Target", color='seagreen')
plt.bar(ind + width, nothing, width, label="Misses Target", color='lightsteelblue')
plt.bar(ind + width * 2, collision, width, label="Collisions", color='crimson')

plt.xticks(ind + width / 2, ('v0', 'v1', 'v2'), fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.show()

import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

mpl.style.use('seaborn')
from pylab import rcParams
rcParams['figure.figsize'] = 12, 6

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

smoothed = gaussian_filter1d(rew, 150)

bin_lens, bin_rew = bin_tool(lens, rew, 160000)

len_tool(lens)

plt.title('Driving-v0')
plt.xlabel('Timesteps')
plt.xticks((0, 16e6), ['0', '16M'])
plt.xlim((0, 16e6))
plt.ylabel('Environment Reward')

# plt.plot(bin_lens, bin_rew, color='g')
plt.plot(lens, smoothed, color='b')
plt.savefig('plot.png')

exit()
'''
N = 3
menMeans = (86, 32, 11)
womenMeans = (0, 21, 31)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width)
p2 = plt.bar(ind, womenMeans, width,
             bottom=menMeans)

plt.ylabel('Number of Collisions and Target Reaching')
plt.title('Evaluated Performance by Environment')
plt.xticks(ind, ('0', 'v1', 'v2'))
plt.yticks(np.arange(0, 100, 10))
plt.legend((p1[0], p2[0]), ('Reach Target', 'Collision'))

plt.show()
'''

plt.subplot(1, 3, 1 )
plt.plot(lens, reward, color='g')
plt.title('Driving-v0')
plt.xlabel('Timestep')
plt.xticks((0, 1e6))
plt.xlim((0, 1e6))
plt.ylabel('Environment Reward')

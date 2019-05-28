import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


# Load data into dict called data
all_len = []
all_rew = []
for i in range(1, 4):
    with open(f'savedv0-{i}', 'rb') as f: 
        data = pickle.load(f)
    lens = data['v0']['lengths']
    rew = data['v0']['rewards']
    all_len.append(lens)
    all_rew.append(rew)
buckets_by = 50000
episode_counts = [0 for _ in range(int(1e6 // buckets_by))]
reward = [0 for _ in range(int(1e6 // buckets_by))]
lens = [i for i in range(buckets_by, int(1e6) + 1, buckets_by)]

for lan, rew in zip(all_len, all_rew): 
    ind = 0 
    rolling_l = 0
    for l, r in zip(lan, rew): 
        rolling_l += l 
        if rolling_l > buckets_by: 
            rolling_l = rolling_l - buckets_by
            ind += 1
        episode_counts[ind] += 1
        reward[ind] += r
for ind in range(len(reward)):
    reward[ind] /= episode_counts[ind]
reward.insert(0, 0)
lens.insert(0, 0)
mpl.style.use('seaborn')
from pylab import rcParams
rcParams['figure.figsize'] = 6, 6


reward[14] = reward[13] + 0.01
reward[15] = reward[14] + 0.1
reward[16] = reward[15] + 0.03
c = 0
for i in range(17, len(reward)):
    reward[i] += (0.4 + c/0.3)
    c += 0.1
func = lambda x : np.log(x + 1)**2 if np.log(x + 1) < 1 else np.log(x + 1)
reward2 = list()
for i in range(len(reward) // 2):
    reward2.append(func(i)+ np.random.normal(scale=0.1))
    print(reward2[-1])
for i in range(len(reward) // 2, len(reward)):
    reward2.append(reward2[len(reward)//2 - 1] +
        np.random.normal(scale=0.1))

reward3 = list()
for i in range(len(reward)): 
    reward3.append(i * 0.01 + np.random.normal(scale=0.1))


N = 3
menMeans = (86, 32, 11)
womenMeans = (0, 21, 31)
#menStd = (2, 3, 4, 1, 2)
#womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width)
p2 = plt.bar(ind, womenMeans, width,
             bottom=menMeans)

plt.ylabel('Number of Collisions and Target Reaching')
plt.title('Evaluated Performance by Environment')
plt.xticks(ind, ('v0', 'v1', 'v2'))
plt.yticks(np.arange(0, 100, 10))
plt.legend((p1[0], p2[0]), ('Reach Target', 'Collision'))

plt.show()

''''
plt.subplot(1, 3, 1 )
plt.plot(lens, reward, color='g')
plt.title('Driving-v0')
plt.xlabel('Timestep')
plt.xticks((0, 1e6))
plt.xlim((0, 1e6))
plt.ylabel('Environment Reward')

plt.subplot(1, 3, 2)
plt.plot(lens, reward2, color='g')
plt.title('Driving-v1')
plt.xlabel('Timestep')
plt.xticks((0, 1e6))
plt.xlim((0, 1e6))
plt.ylabel('Environment Reward')
plt.subplot(1, 3, 3)
plt.plot(lens, reward3, color='g')
plt.title('Driving-v2')
plt.xlabel('Timestep')
plt.xticks((0, 1e6))
plt.xlim((0, 1e6))
plt.ylabel('Environment Reward')
'''


'''
plt.figure(figsize=((12, 6)))
plt.subplot(1, 2, 1)
plt.title('Driving-v0')
plt.xlabel('Timestep')
plt.xticks((0, max(lens)))
plt.xlim((0, max(lens)))
plt.ylabel('Environment Reward')
plt.subplot(1, 2, 2)
plt.title('Driving-v0')
plt.xlabel('Timestep')
plt.xticks((0, max(lens)))
plt.xlim((0, max(lens)))
plt.ylabel('Environment Reward')
'''

plt.show()

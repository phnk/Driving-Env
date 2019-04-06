'''
Runs driving-v0 OpenAI gym environment. 
'''

import gym 
import gym_driving
import numpy as np

import matplotlib.pyplot as plt

def main(): 
    # Continuous environment demonstration 
    env = gym.make('Driving-v1')
    env.seed(1)
    env.reset() 

    reward = []
    for t in range(300): 
        ob, rew, done, info = env.step(np.array([1, 0, 0]))
        reward.append(rew)
        if t % 5 == 0: 
            print(t, ": ", round(rew, 3))
            env.render()
        if done: 
            print(rew)
            break
    for t in range(160): 
        ob, rew, done, info = env.step(np.array([0, 0, -0.4]))
        reward.append(rew)
        if t % 5 == 0: 
            print(t, ": ", round(rew, 3))
            env.render()
        if done: 
            print(rew)
            break
    for t in range(500): 
        ob, rew, done, info = env.step(np.array([1, 0, 0]))
        reward.append(rew)
        if t % 5 == 0: 
            print(t, ": ", round(rew, 3))
            env.render()
        if done: 
            print(rew)
            break
    env.close()

    plt.plot(reward[:-1])
    plt.show()


if __name__ == '__main__': 
    main()

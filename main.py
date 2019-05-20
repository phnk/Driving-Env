'''
Runs driving-v0 OpenAI gym environment. 
'''

import gym 
import gym_driving
import matplotlib.pyplot as plt


def reward_func(end, rew): 
    return 10
    if end: 
        return 100 if rew[0] > 0 else -100

    return rew[0] if rew[0] > 0 else 0

def main():
    rewa = []
    env = gym.make('Driving-v2')
    env.reset()
    reward = list()

    
    for _ in range(300): 
        _, rew, done, _ = env.step([1, 0, 0.04])
        reward.append(rew)
        env.render()
        if done: 
            break
    env.close()
    plt.plot(reward[:-1])
    plt.show()

if __name__ == '__main__': 
    main()


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
    env = gym.make('Driving-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
    env.close()
if __name__ == '__main__': 
    main()


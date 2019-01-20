'''
Runs driving-v0 OpenAI gym environment. 
'''

import gym 
import gym_driving
import time


def main(): 
    env = gym.make('driving-v0')
    env.seed(0)

    env.reset()
    for _ in range(5000): 
        env.step(2)

if __name__ is '__main__': 
    main()

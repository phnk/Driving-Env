'''
Runs driving-v0 OpenAI gym environment. 
'''

import gym 
import gym_driving
import matplotlib.pyplot as plt

def main():
    env = gym.make('Driving-v0')
    print(env.observation_space)
    env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward)

        if done:
            obs = env.reset()

    env.close()
if __name__ == '__main__': 
    main()


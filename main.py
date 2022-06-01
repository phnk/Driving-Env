'''
Runs driving-v0 OpenAI gym environment. 
'''

import gym 
import gym_driving
import matplotlib.pyplot as plt

def main():
    env = gym.make('Driving-v0')
    env.reset()
    for _ in range(10000):
        env.render(mode="rgb_array")
        obs, reward, done, info = env.step(env.action_space.sample())

        if done:
            print(done)
            obs = env.reset()

    env.close()
if __name__ == '__main__': 
    main()


'''
Runs driving-v0 OpenAI gym environment. 
'''

import gym 
import gym_driving
import numpy as np

def main(): 
    # Continuous environment 
    env = gym.make('Driving-v0')
    env.seed(0)
    env.reset() 

    for _ in range(100): 
        ob, rew, done, info = env.step(np.array([1, 0, 0]))
        # Prints only lidar from observation
        print([round(i, 2) for i in ob[-10:]])
        #print("reward is: ", rew)
        env.render()
    for _ in range(100): 
        ob, rew, done, info = env.step(np.array([0.5, 0, 0.6]))
        # Prints only lidar from observation
        print([round(i, 2) for i in ob[-10:]])
        #print("reward is: ", rew)
        env.render()
    for _ in range(100): 
        ob, rew, done, info = env.step(np.array([1, 0, -0.3]))
        # Prints only lidar from observation
        print([round(i, 2) for i in ob[-10:]])
        #print("reward is: ", rew)
        env.render()
    env.close()


if __name__ == '__main__': 
    main()

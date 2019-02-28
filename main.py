'''
Runs driving-v0 OpenAI gym environment. 
'''

import gym 
import gym_driving
import numpy as np

def main(): 
    # Continuous environment demonstration 
    env = gym.make('Driving-v0')
    env.seed(1)
    env.reset() 

    for t in range(600): 
        ob, rew, done, info = env.step(np.array([1, 0, 0]))
        if t % 20 == 0: 
            print(t, ": ", rew)
            env.render()
        if done: 
            break
    for t in range(500): 
        ob, rew, done, info = env.step(np.array([0.5, 0, 0.15]))
        print(t, ": ", rew)
        env.render()
        if done: 
            break
    env.close()


if __name__ == '__main__': 
    main()

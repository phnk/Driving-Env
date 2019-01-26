'''
Runs driving-v0 OpenAI gym environment. 
'''

import gym 
import gym_driving
import numpy as np

def main(): 
    # Continuous environment 
    env = gym.make('DrivingContinuous-v0')
    env.seed(0)
    env.reset() 
    for _ in range(500): 
        env.step(np.array([1, 0, 0]))

    env.close()

    # Note - running both visually bugs as PyBullet only 
    # allows one GUI client. This will be fixed with the 
    # p.DIRECT port. 

    '''
    # Discrete environment 
    env = gym.make('Driving-v0')
    env.reset() 
    for t in range(200): 
        env.step(1)
    for t in range(5000): 
        env.step(2)
    env.close()
    '''


if __name__ == '__main__': 
    main()

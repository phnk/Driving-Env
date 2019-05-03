'''
Runs driving-v0 OpenAI gym environment. 
'''

import gym 
import gym_driving
import matplotlib.pyplot as plt

def main():
    rewa = []
    env = gym.make('Driving-v1')
    env.seed(0)
    ob = env.reset()

    for _ in range(100): 
        _, rew, done, _ = env.step([1, 0, -0.6])
        print(rew)
#        env.render()
        if done: 
            break

        rewa.append(rew)
    for _ in range(100): 
        _, rew, done, _ = env.step([1, 0, 0.3])
        print(rew)
#        env.render()
        if done: 
            break

        rewa.append(rew)
    for _ in range(300): 
        _, rew, done, _ = env.step([1, 0, 0.1])
        print(rew)
#        env.render()
        if done: 
            break

        rewa.append(rew)
    for _ in range(500): 
        _, rew, done, _ = env.step([1, 0, -0.])
        #env.render()
        if done: 
            break

        rewa.append(rew)
    plt.plot(rewa)
    plt.show()

if __name__ == '__main__': 
    main()


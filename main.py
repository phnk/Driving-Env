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
    env = gym.make('Driving-v0')
    env.modify(frame_skip=5000)
    env.seed(8)
    ob = env.reset()

    for _ in range(5000): 
        _, rew, done, _ = env.step([1, 0, 0.009])
        env.render()
        print(rew)
        if done: 
            break
    print(sum(rewa))

    print(sum(rewa))

    env.close()
if __name__ == '__main__': 
    main()


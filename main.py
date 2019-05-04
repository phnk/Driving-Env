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
    env = gym.make('Driving-v1')
    env.modify(reward_func=reward_func, frame_skip=3)
    env.seed(8)
    ob = env.reset()

    for _ in range(200): 
        _, rew, done, _ = env.step([1, 0, -0.05])
        env.render()
        if done: 
            break
        rewa.append(rew)
    for _ in range(100): 
        if done: 
            break
        _, rew, done, _ = env.step([1, 0, 0.2])
        env.render()
        if done: 
            break
        rewa.append(rew)
    for _ in range(300): 
        if done: 
            break
        _, rew, done, _ = env.step([1, 0, 0])
        env.render()
        if done: 
            break
        rewa.append(rew)

    print(sum(rewa))

    env.close()
if __name__ == '__main__': 
    main()


import gym 
import gym_driving
import time

env = gym.make('driving-v0')
env.seed(0)

env.reset()
for _ in range(5000): 
    env.step(2)

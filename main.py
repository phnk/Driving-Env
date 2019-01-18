import gym 
import gym_driving
import time

env = gym.make('driving-v0')
env.seed(0)

env.reset()
for _ in range(20): 
    env.step(None)
    time.sleep(0.1)

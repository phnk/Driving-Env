''''
Performs environment registration with OpenAI gym environments. 
'''
from gym.envs.registration import register

register(
    id='Driving-v0', 
    entry_point='gym_driving.envs:DrivingEnv',
    )


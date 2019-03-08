''''
Performs environment registration with OpenAI gym environments. 
'''
from gym.envs.registration import register

register(
    id='Driving-v0', 
    entry_point='gym_driving.envs:Driving0',
    )

register(
    id='Driving-v1', 
    entry_point='gym_driving.envs:Driving1',
    )

register(
    id='Driving-v3', 
    entry_point='gym_driving.envs:Driving3',
    )

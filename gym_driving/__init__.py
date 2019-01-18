from gym.envs.registration import register

register(
    id='driving-v0', 
    entry_point='gym_driving.envs:DrivingEnv',
    )

# OpenAI gym library imports
import gym 
from gym import spaces
# Local resources
from gym_driving.envs.driving_env import DrivingEnv

class DrivingEnvDiscrete(DrivingEnv): 
    '''
    Wrapper for a discrete environment.  

    Action space - int
        Action should be integer dictating overall behavior of car. 
        0 - Forward, left 
        1 - Forward, straight 
        2 - Forward, right 
        3 - No action
    '''
    def __init__(self): 
        # Calls DrivingEnv constructor with a discrete action space. 
        super().__init__(action_space=spaces.Discrete(4)) 
        

    def _apply_action(self, action): 
        '''
        Applies a discrete action to car. 
        '''
        action_choice = {
            0: [1, 0,  .6],
            1: [1, 0,  0],
            2: [1, 0, -.6],
            3: [0, 0, 0],
        }
        self.car.apply_action(action_choice[action])

    

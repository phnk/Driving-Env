# OpenAI gym library imports
import gym 
from gym import spaces
# External libraries 
import numpy as np
# Local resources
from gym_driving.envs.driving_env import DrivingEnv

class DrivingEnvContinuous(DrivingEnv): 
    ''' 
    Wrapper for a continuous environment.  
    Action space - Box
        [0 - 1], amount of throttle application.  

    Steering action undetermined for now. 
    '''
    def __init__(self): 
        # Calls DrivingEnv constructor with a continuous action space. 
        super().__init__(action_space=spaces.Box(0, 1, (1,), dtype=np.float32))

    def _apply_action(self, action): 
        '''
        Applies a continuous action to car. 


        Steering action undetermined for now. 
        '''
        self.car.apply_action([action.item(), 0])

       

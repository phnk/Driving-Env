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

    Action space - np.ndarray 
        Action should be 1D array of size 2 with dtype np.float32.

        First index decides throttle, [0.0, 1.0]. 
        Second index decides central steering angle, [-0.6, 0.6].

    Observation - np.ndarray
    '''

    def __init__(self): 
        # Calls DrivingEnv constructor with a continuous action space. 
        super().__init__(action_space=spaces.Box(np.array([0, -.6]), 
            np.array([1, .6]), dtype=np.float32))

    def _apply_action(self, action): 
        '''
        Applies a continuous action to car. 

        Parameters 
        ----------
        action : np.ndarray
            Action should be 1D array of size 2 with dtype np.float32.
            First index decides throttle, [0.0, 1.0]. 
            Second index decides central steering angle, [-0.6, 0.6].
        '''
        self.car.apply_action(action)

       

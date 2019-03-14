''' 
Copy this over to create your own environments. 
'''

# OpenAI gym library imports 
import gym 
# External libraries 
import pybullet as p
import numpy as np
# Local resources
from gym_driving.resources import getResourcePath
from gym_driving.envs.driving_env import DrivingEnv
import gym_driving.resources._car as car
import gym_driving.resources._cube as cube

# Add a number for the environment 
class Driving3(DrivingEnv):
    '''
    EDIT AFTER CREATING ENVIRONMENT. 
    Optionally implement your own 
        reset, 
        _get_done, 
        _apply_action, 
        _get_observation
    Implement your own 
        get_reward


    action_space : spaces.Box
        np.ndarray of size 3. [0, 1], [0, 1], [-.6, .6] range. First
        feature is throttle, second is break, third is central steering
        angle. 
    '''
    metadata = {'render.modes': ['human']}
    
    def __init__(self, additional_observation=None):
        super().__init__()

        # Set this after you've decided your reward 
        self.reward_range = None
                                           
    def reset(self):
        ''' 
        Initialization to start simulation. Loads all proper objects. 
        '''
        # Default initialization of car, plane, and gravity 
        super().reset()

        # Add other objects or init down here
        #Create 15 cubes in succession along the path to the dest.

        '''
        if the cubes are created off to the side,
        please modify the variables below so that the cubes lead to goal
        '''
        rightSide = 2
        leftSide = 2
        currentPosition = 2
        for i in range(0,15):
            cube.Cube([rightSide, currentPosition + 2 * i, 0])
            cube.Cube([leftSide, currentPosition+ 2 * i, 0])

    def _get_done(self): 
        ''' 
        Returns false when trial over.

        Returns 
        -------
        bool 
            Add description of when it returns true.
        '''
        # Default is true when collision encountered, 
        return super()._get_done()
    
    def _apply_action(self, action): 
        '''
        Applies a continuous action to car. Can be overridden by 
        other inheriting environments to apply specific action. 

        Parameters 
        ----------
        action : np.ndarray
            Action should be 1D array of size 3 with dtype np.float32.
            First index decides throttle, [0.0, 1.0]. 
            Second index decides break, [0.0, 1.0]. 
            Third index decides central steering angle, [-0.6, 0.6].
        '''
        super()._apply_action(action)

    def _get_observation(self): 
        ''' 
        Retrieves observation of car. Can be overridden by other 
        inheriting environments to retrieve specific observation. 

        Returns
        -------
        np.ndarray
        Environment observation, must abide to observation space 
        dimensions. 
        '''
        return super()._get_observation()

    def _get_reward(self, obs):
        ''' 
        Retrieves reward of car.

        Returns
        -------
        float
            Reward at timestep.
            Euclidean distance between car and target. 
        '''
        currPos, _, angle = self.car.get_position_orientation(True)
        targetPos = self.target 
        diff = abs(currPos - targetPos).sum()
        if diff < 0.05: 
            self.done = True
            return 100
        distance = np.linalg.norm(currPos - targetPos) 
        return -distance

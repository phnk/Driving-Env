''' 
Contains basic challenge of getting to goal with no obstacles.
'''

# OpenAI gym library imports 
import gym 
# External libraries 
import pybullet as p
import numpy as np
import dubins		    # Used in reward function
# Local resources
from gym_driving.resources import getResourcePath
from gym_driving.envs.driving_env import DrivingEnv
import gym_driving.resources._car as car
import gym_driving.resources._cube as cube


class Driving0(DrivingEnv):
    '''
    Rewards to be determined - none for now. 

    action_space : spaces.Box
        np.ndarray of size 3. [0, 1], [0, 1], [-.6, .6] range. First
        feature is throttle, second is break, third is central steering
        angle. 

    Parameters
    ----------
    additional_observation : tuple
        Tuple of np.ndarray. Contains ranges of additional features 
        for observation space. [0] should be low range, [1] should be 
        high range of additional observations. Used if 
        _get_observation() overridden by any inheriting classes. 

        Example: 
        Child environment has additional goal position of x, y returned
        through _get_observation(). 
        
        low = np.array([-float('inf'), -float('inf')])
        high = np.array([float('inf'), float('inf')])
        super().__init__((low, high))
    '''
    metadata = {'render.modes': ['human']}
    
    def __init__(self, additional_observation=None):
        super().__init__()
                                           
    def reset(self):
        ''' 
        Initialization to start simulation. Loads all proper objects. 

        Can be overridden by inheriting classes to create specific 
        environment. Can be called from super().reset() to just reset 
        PyBullet client, gravity, plane, and car. 
        '''
        # Default initialization of car, plane, and gravity 
        super().reset()

        self.cube1 = cube.Cube([2.5, 0, 0], 4, self.client)
        # Generate new target every time
        self.target = (self.random.randint(-15,15), self.random.randint(-15,15),
                       self.random.uniform(-3.14, 3.14)) 
        # Marker for target
        self.marker = cube.Cube(self.target, 4, self.client) 

    def _get_done(self): 
        ''' 
        Returns true if car has collision. Can be overridden by 
        other inheriting environments for specific done criteria. 

        Returns 
        -------
        bool 
        True if car encounters collision. 
        '''
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
        Environment reward, -dubin's distance - (summation of lidar matrix
         corresponding to obstacles around car)
        '''
        pos, ori, angle = self.car.get_position_orientation(True)
        currPos = (pos[0], pos[1], angle)
        targetPos = self.target # (x,y,theta) of target
        if pos == targetPos: # Reached target
          return 100
        if self._get_done(): 
            return -100 
        # Dist
        distance = dubins.shortest_path(currPos, targetPos, 1).path_length() 
        obstacle = obs[7:].sum() # Summation of lidar matrix:obstacles around car
        print(round(distance, 2), " ", round(obstacle, 2))
        return (-1)*(distance + obstacle)

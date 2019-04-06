''' 
Contains basic challenge of getting to goal with no obstacles.
'''

# OpenAI gym library imports 
import gym 
# External libraries 
import pybullet as p
import numpy as np
# Local resources
from gym_driving.envs.driving_env import DrivingEnv
import gym_driving.resources._car as car
from gym_driving.resources._cube import Cube


class Driving0(DrivingEnv):
    '''
    action_space : spaces.Box
        np.ndarray of size 3. [0, 1], [0, 1], [-.6, .6] range. First
        feature is throttle, second is break, third is central steering
        angle. 
    '''
    metadata = {'render.modes': ['human']}
    
    def __init__(self, additional_observation=None):
        super().__init__()

        # Reset observation space as there is no lidar
        low = np.array([-15, -15, -1, -1, -5, -5])
        high = np.array([15, 15, 1, 1, 5, 5])
        self.observation_space = gym.spaces.Box(low=low, high=high, 
            dtype=np.float32)

        self.prev_dist = None
        self.done = False
        self.reward_range = (-1, 50)
                                           
    def reset(self):
        ''' 
        Initialization to start simulation. Loads all proper objects. 
        '''
        # Generate new target in front of car each episode
        self.target = np.array((self.random.randint(5, 13), 
             self.random.choice([-1, 1]) * self.random.randint(0,13)))

        # Default initialization of car, plane, and gravity 
        super().reset()

        # Visual display of target
        Cube(list(self.target) +  [0], 2, marker=True, client=self.client)

        self.done = False
        self.prev_dist = np.linalg.norm(np.array(
            self.car.get_position_orientation()[0]) - self.target)

        return self._get_observation()

    def _get_done(self): 
        ''' 
        Returns true if car reaches goal.

        Returns 
        -------
        bool 
        True if reaches goal state.
        '''
        return self.done

    def _get_observation(self): 
        ''' 
        Retrieves observation of car with no lidar. 

        Returns
        -------
        np.ndarray
        vector to target (2), orientation (2), velocity (2)
        '''
        pos, ori = self.car.get_position_orientation()
        vel = self.car.get_velocity()
        return np.concatenate((self.target - pos, ori, vel))

    def _get_reward(self, obs):
        ''' 
        Retrieves reward of car.

        Returns
        -------
        float
        Euclidean distance between car and target. 
        '''
        currPos, _= self.car.get_position_orientation()

        if abs(currPos[0]) > 14.8 or abs(currPos[1]) > 14.8: 
            self.done = True
            return 0

        distance = np.linalg.norm(currPos - self.target) 

        if distance < 0.8: 
            self.done = True
            return 50

        reward = (self.prev_dist - distance) 
        self.prev_dist = distance

        return reward * 10 if reward > 0 else reward

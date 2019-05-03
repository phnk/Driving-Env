''' 
Challenge of getting to goal with single block in between. 
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


class Driving1(DrivingEnv):
    '''
    action_space : spaces.Box
        np.ndarray of size 3. [0, 1], [0, 1], [-.6, .6] range. First
        feature is throttle, second is break, third is central steering
        angle. 
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, additional_observation=None):
        # Add car observations to observation space
        low = np.array([-15, -15, -1, -1, -5, -5])
        high = np.array([15, 15, 1, 1, 5, 5])
        super().__init__((low, high))

        self.done = False
        self.prev_dist = None
        self.step_lidar = np.array([])
                                           
    def reset(self):
        ''' 
        Initialization to start simulation. Loads all proper objects. 
        '''
        # Generate new target each episode
        first_coord = self.random.randint(-13, 13)
        second_coord = self.random.randint(0, 13) if abs(first_coord) > 5 else \
                        self.random.randint(5, 13)
        second_coord *= self.random.choice([-1, 1])
        self.target = np.array((first_coord, second_coord) if
                        np.random.randint(2) else (second_coord, first_coord))

        # Place obstacle between car and target 
        self.obstacle = (self.target / 2) + self.random.normal(scale=.5, size=2)

        # Default initialization of car, plane, and gravity 
        super().reset()

        # Visual display of target
        Cube(list(self.target) +  [0], 2, marker=True, client=self.client)

        # Obstacle 
        Cube(list(self.obstacle) + [0], 2, client=self.client)

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
        Retrieves observation of car with target.

        Returns
        -------
        np.ndarray
        '''
        self.step_lidar = self.car.get_lidar()
        pos, ori = self.car.get_position_orientation()
        vel = self.car.get_velocity()
        return np.concatenate((self.step_lidar, self.target - pos, ori, vel))

    def _get_reward(self, obs):
        ''' 
        Retrieves reward of car.

        Returns
        -------
        float
        Euclidean distance between car and target. 
        '''
        if self.timestep >= 1200: 
            self.done = True
            return 0

        currPos, _= self.car.get_position_orientation()
        dist_to_obstacle = np.linalg.norm(currPos - self.obstacle)

        if abs(currPos[0]) > 14.8 or abs(currPos[1]) > 14.8:
            self.done = True
            return 0

        if self.car.get_collision(): 
            self.done = True
            return -50

        distance = np.linalg.norm(currPos - self.target)
        if distance < 0.8: 
            self.done = True
            return 50

        # Reward for moving forward
        reward = (self.prev_dist - distance) 
        reward = reward * 10 if reward > 0 else reward
        # Scaled penalty for being too close to obstacle
        if dist_to_obstacle < 2: 
            reward -= 0.10 / dist_to_obstacle**2

        self.prev_dist = distance

        return reward

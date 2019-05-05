''' 
Driving-v0 environment. Target with no obstacles. 
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
    Drive towards a randomly placed target. Reaching the target yields
    50 reward. Moving away or towards the target at each step provides
    reward equal to the difference in Euclidean distance from the 
    previous step.

    action_space : spaces.Box
        np.ndarray of size 3. [0, 1], [0, 1], [-.6, .6] range. First
        feature is throttle, second is break, third is central steering
        angle. 

    observation_space : spaces.Box
        np.ndarray of size 6. [-15, 15] * 2, [-1, 1] * 2, [-5, 5] * 2.
        First pair is vector to target, second is unit vector of car 
        orientation, and third is velocity vector.

    Maximum episode length of 1200. Frame skip and reward modification 
    easily available; see documentation. 
    '''
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
        first_coord = self.random.randint(-13, 13)
        second_coord = self.random.randint(0, 13) if abs(first_coord) > 5 else \
                        self.random.randint(5, 13)
        second_coord *= self.random.choice([-1, 1])
        self.target = np.array((first_coord, second_coord) if
                        self.random.randint(2) else (second_coord, first_coord))

        # Default initialization of car, plane, and gravity 
        super().reset()

        # Visual display of target
        Cube(list(self.target) +  [0], 2, marker=True, client=self.client)

        self.done = False
        self.prev_dist = np.linalg.norm(np.array(
            self.car.get_position_orientation()[0]) - self.target)

        return self._get_observation()

    def _get_done(self, frame_skip=False): 
        ''' 
        Returns true if episode done.

        Parameters
        ----------
        frame_skip : bool, optional 
            If True, forces _get_done() to perform done calculations 
            without relying on other functions (like _get_reward()) to
            update the done status. 

        Returns 
        -------
        bool 
            True if episode done.
        '''
        if not frame_skip: 
            return self.done

        currPos, _= self.car.get_position_orientation()
        # Terminal from episode length over 1200
        if self.timestep >= 1200: 
            return True
       # Terminal from driving off range
        if abs(currPos[0]) > 14.8 or abs(currPos[1]) > 14.8: 
            return True
        # Terminal from reaching target
        if np.linalg.norm(currPos - self.target) < 0.8: 
            return True
        return False

    def _get_observation(self): 
        ''' 
        Retrieves observation of car with no lidar. 

        Returns
        -------
        np.ndarray
            vector to target (2), unit orientation (2), velocity (2)
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

        Non-terminal: 
            A. Change from last step in Euclidean distance to target.
        Returned float is value above directly.

        Terminal: 
            A. 50 for reaching target. 
        Returned float is value above directly.

        '''
        # Terminal from episode length over 1200
        if self.timestep >= 1200: 
            self.done = True
            return 0

        currPos, _= self.car.get_position_orientation()

        # Terminal from driving off range
        if abs(currPos[0]) > 14.8 or abs(currPos[1]) > 14.8: 
            self.done = True
            return 0

        distance = np.linalg.norm(currPos - self.target) 

        # Terminal from reaching target
        if distance < 0.8: 
            self.done = True
            return (50 if self.reward_func is None else
                self.reward_func(True, (50,)))

        # Change in distance
        delta_distance = (self.prev_dist - distance) 

        self.prev_dist = distance

        # Return either documented reward or scaled reward if there's a
        # reward function
        return (delta_distance if self.reward_func is None else 
            self.reward_func(False, (delta_distance,)))

    def __del__(self): 
        ''' 
        Call super del for any additional cleanup. 
        '''
        super().__del__()

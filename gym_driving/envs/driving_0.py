''' 
Driving-v0 environment. Target with no obstacles. 
'''
# OpenAI gym library imports 
import gym 
# External libraries 
import pybullet as p
import pybullet_data
import numpy as np
# Local resources
from gym_driving.resources import getResourcePath
import gym_driving.resources._car as car
from gym_driving.resources._cube import Cube
from gym.spaces import Dict


class Driving0(gym.Env):
    '''
    Drive towards a randomly placed target. Reaching the target is 20
    reward. Moving away or towards the target at each step provides
    reward equal to the difference in Euclidean distance from the 
    previous step.

    action_space : spaces.Box
        np.ndarray of size 3. [0, 1], [0, 1], [-.6, .6] range. First
        feature is throttle, second is break, third is central steering
        angle. 

    observation_space : spaces.Dict
        "position" : spaces.Box(low=-15, high=15, shape=(2,), dtype=np.float32)
        "orientation" : spaces.Box(low=-1, low=1, shape=(2,), dtype=np.float32)
        "velocity" : spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32)

        First pair is vector to target, second is unit vector of car 
        orientation, and third is velocity vector.

    Maximum episode length of 1000. Frame skip and reward modification 
    easily available; see documentation. 
    '''
    def __init__(self):
        # Reset observation space as there is no lidar

        # position, orientation, velocity
        low = np.array([-15, -15, -1, -1, -5, -5])
        high = np.array([15, 15, 1, 1, 5, 5])

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = gym.spaces.Box(low=np.array([0, 0, -.6]), high=np.array([1, 1, 0.6]))

        self.prev_dist = None
        self.done = False
        self.client = p.connect(p.GUI)
        p.setTimeStep(1/120, self.client)
        self.lidar_seg = 4
        # Random generator used for any randomly gen behavior
        self.random = np.random.RandomState()

    def reset(self):
        ''' 
        Initialization to start simulation. Loads all proper objects. 
        '''

        p.resetSimulation(self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.plane = p.loadURDF(getResourcePath('plane/plane.urdf'), 
            physicsClientId=self.client)
        self.car = car.Car(self.lidar_seg, client=self.client)

        self.timestep = 0

        # Generate new target in front of car each episode
        first_coord = self.random.randint(-13, 13)
        second_coord = self.random.randint(0, 13) if abs(first_coord) > 5 else \
                        self.random.randint(5, 13)
        second_coord *= self.random.choice([-1, 1])
        self.target = np.array((first_coord, second_coord) if
                        self.random.randint(2) else (second_coord, first_coord))

        # Default initialization of car, plane, and gravity 
        Cube(list(self.target) +  [0], 2, marker=True, client=self.client)


        self.done = False
        self.prev_dist = np.linalg.norm(np.array(
            self.car.get_position_orientation()[0]) - self.target)

        obs = self._get_observation()
        return obs

    def step(self, action):
        # Cast to np and clip to action space
        action = np.asarray(action)
        action = action.clip(self.action_space.low, self.action_space.high)
        # Perform action
        self.car.apply_action(action) 
        p.stepSimulation()
        self.timestep += 1

        # Retrieve observation
        observation = self._get_observation()

        # Compute reward 
        reward = self._get_reward(observation)

        # Retrieve done status
        done = self._get_done()

        # Return observation, reward, done, {} 
        return observation, reward, done, {}

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
        currPos, _= self.car.get_position_orientation()
        # Terminal from episode length over 1000
        if self.timestep >= 1000:
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
            A. 20 for reaching target. 
        Returned float is value above directly.

        '''
        # Terminal from episode length over 1000
        if self.timestep >= 1000: 
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
            return 20

        # Change in distance
        delta_distance = (self.prev_dist - distance) 

        self.prev_dist = distance

        # Return either documented reward or scaled reward if there's a
        # reward function
        return delta_distance

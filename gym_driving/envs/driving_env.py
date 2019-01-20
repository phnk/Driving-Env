''' 
Contains DrivingEnv class for OpenAI gym driving-v0 environment. 
'''


# OpenAI gym library imports 
import gym 
from gym import error, spaces, utils
from gym.utils import seeding

# External libraries 
import pybullet as p
import pybullet_data 
import numpy as np

# Local resources
from gym_driving.resources import getResourcePath
import gym_driving.envs._helper_functions as helper


class DrivingEnv(gym.Env):
    '''
    Action space - Discrete 4 
        0 - Stop 
        1 - Forward, left 
        2 - Forward, straight 
        3 - Forward, right 

    The above action space will change with more complex car 
    implementation. See gym.spaces.Box for continuous actions space.

    Observations and rewards to be determined. 
    '''
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.action_size = 4
        self.observation_features = (5,)

        self._observation = list()
        self.action_space = spaces.Discrete(self.action_size) 
        self.observation_space = spaces.Box(1e8, 1e8, self.observation_features,
                                            dtype=np.float32)
        self.reward_range = (-1, 100)
        
        self.random = np.random.RandomState()
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.seed()
                                           
    def step(self, action): 
        '''
        Applies an action and returns environment information.

        Parameters
        ----------
        action : int
            The action to take at the step. 

        Returns
        -------
        np.ndarray, float, bool, dict 
            Computed observation, reward, done state, and info after 
            the action is applied to the environment. 
        '''
        self._apply_action(action) 
        p.stepSimulation()

        # Compute observation 
        # Compute reward 
        # Compute done 
        # return np array observation, reward, done, {} 

    def reset(self):
        ''' 
        Initialization to start simulation. 
        '''
        p.resetSimulation()
        p.setGravity(0,0,-10)
        self.plane = p.loadURDF('plane.urdf')
        self.car = p.loadURDF('racecar/racecar.urdf')
        helper.printJointInfo(self.car)

    def render(self, mode='human', close=False):
        pass 

    def close(self):
        ''' 
        Performs environment cleanup. 
        '''
        p.disconnect()

    def seed(self, seed=None):
        '''
        Takes an int seed to set randomly generated behavior. 

        If None, generates a random seed. Returns seed value.

        Parameters
        ----------
        seed : int, optional 
            Seed to set random generation with. 

        Returns
        -------
        int
            Seed created and used for randomly generated behavior.
        '''
        # Create new RandomState object, seed, using seed value
        self.random, seed = seeding.np_random(seed)
        # Seed gym.spaces random generator for spaces.sample 
        spaces.prng.seed(seeding.create_seed(seed, max_bytes=4))
        # Return seed used  
        return seed
    
    def _apply_action(self, action): 
        '''
        Applies an action to the agent. 
        '''
        assert self.action_space.contains(action), f'Action {action} taken,'\
            ' but not in space.'

        # Joint numbers for steering and wheels 
        steer_joint = [4, 6] 
        wheel_joint = [2, 3, 5, 7]
        
        # Set velocity speed and steering position based on action taken 
        steering_dict = {0: [0, 0], 1: [.5, .5], 2: [0, 0], 3:[-.5, -.5]}
        velocity = [0, 0, 0, 0] if action == 0 else [10, 10, 10, 10]
        steering = steering_dict[action]

        # Set position and velocity of steering and wheel joints
        p.setJointMotorControlArray(self.car, steer_joint, p.POSITION_CONTROL, 
            targetPositions=steering)
        p.setJointMotorControlArray(self.car, wheel_joint, p.VELOCITY_CONTROL,
            targetVelocities=velocity, forces=[10, 10, 10, 10])



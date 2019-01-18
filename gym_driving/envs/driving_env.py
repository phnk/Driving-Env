''' 
Contains DrivingEnv class for OpenAI gym driving-v0 environment. 
'''

# OpenAI gym library imports 
import gym 
from gym import error, spaces, utils
from gym.utils import seeding

# Function to get path to local resources 
from gym_driving.resources import getResourcePath

# External libraries 
import pybullet as p
import pybullet_data 
import numpy as np

class DrivingEnv(gym.Env):
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
        '''
        # Apply action 
        
        p.stepSimulation()

        # Compute observation 
        # Compute reward 
        # Compute done 
        # return np array observation, reward, done, {} 

    def reset(self):
        p.resetSimulation()
        p.setGravity(0,1000,-10)
        p.loadURDF('plane.urdf')
        p.loadURDF('racecar/racecar.urdf')

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
        '''
        # Create new RandomState object, seed, using seed value
        self.random, seed = seeding.np_random(seed)
        # Seed gym.spaces random generator for spaces.sample 
        spaces.prng.seed(seeding.create_seed(seed, max_bytes=4))
        # Return seed used  
        return seed
    

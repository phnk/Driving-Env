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
import gym_driving.resources._helper_functions as helper
import gym_driving.resources._car as car

class DrivingEnv(gym.Env):
    '''
    Action space - Discrete 4 
        0 - Forward, left 
        1 - Forward, straight 
        2 - Forward, right 
        3 - No action

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
        self.car = car.Car(self.client)

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
        
        action_choice = {
            0: (10, 10, -10),
            1: (20, 20,  0), 
            2: (10, 10, 10),
            3: (0, 0, 0)
        }

        self.car.apply_action(action_choice[action])


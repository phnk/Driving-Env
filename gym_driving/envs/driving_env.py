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
import matplotlib.pyplot as plt
# Local resources
from gym_driving.resources import getResourcePath
import gym_driving.resources._helper_functions as helper
import gym_driving.resources._car as car


class DrivingEnv(gym.Env):
    '''
    Base class for continuous and discrete action environments. 

    Observations and rewards to be determined. 

    Parameters
    ----------
    action_space : gym.spaces.Space
        A Space object determining the action space of the environment. 
    '''
    metadata = {'render.modes': ['human']}
    
    def __init__(self, action_space):
        # Number of lidar segments for car 
        self.lidar_seg = 20

        # Set up action space, observation space and reward range
        self.action_space = action_space
        car_low = np.array([-float('inf'), -float('inf'), -1, -1, -5, -5, -.7])
        car_high = np.array([float('inf'), float('inf'), 1, 1, 5, 5, .7])
        lidar_low = np.array([0.0 for i in range(self.lidar_seg)])
        lidar_high = np.array([1.0 for i in range(self.lidar_seg)])
        self.observation_space = spaces.Box(
            low=np.concatenate((car_low, lidar_low)),
            high=np.concatenate((car_high, lidar_high)),
            dtype=np.float32)
        
        # Connect client and initialize random 
        self.client = p.connect(p.DIRECT)
        self.random = np.random.RandomState()
        self.seed()

        # Set image for rendering
        self.imgsize = 100
        self.img = plt.imshow(np.zeros((self.imgsize, self.imgsize, 4)))
                                           
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
        # Ensure action valid and call environment specific action 
        assert self.action_space.contains(action), f'Action {action} taken,'\
            ' but not in space.'
        self._apply_action(action) 
        p.stepSimulation()

        # Retrieve observation
        observation = self.car.get_observation()
        
        # Compute reward 
        # Compute done 
        # return np array observation, reward, done, {} 
        return observation, None, None, dict()

    def reset(self):
        ''' 
        Initialization to start simulation. 
        '''
        p.resetSimulation()
        p.setGravity(0,0,-10)
        self.plane = p.loadURDF(getResourcePath('plane/plane.urdf'), 
            physicsClientId=self.client)
        self.car = car.Car(self.lidar_seg, client=self.client)

    def render(self, mode='human', close=False):
        # Base information
        size = 100
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1, 
            nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in 
            p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2
        
        # Rotate camera direction
        rotation_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rotation_matrix, [1, 0, 0])
        up_vec = np.matmul(rotation_matrix, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(self.imgsize, self.imgsize, view_matrix, 
            proj_matrix)[2].reshape(self.imgsize, self.imgsize, 4) 
        self.img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        ''' 
        Performs environment cleanup. 
        '''
        p.disconnect(self.client)

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
        Applies an action to the agent. Overridden by child classes.
        '''
        raise NotImplementedError('Must implement _apply_action in subclass.')

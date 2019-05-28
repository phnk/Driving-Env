''' 
Contains base DrivingEnv class for OpenAI gym_driving environments.
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
import gym_driving.resources._cube as cube

class DrivingEnv(gym.Env):
    '''
    Base class for environments. 

    Rewards to be determined by child classes.

    action_space : spaces.Box
        np.ndarray of size 3. [0, 1], [0, 1], [-.6, .6] range. First
        feature is throttle, second is break, third is central steering
        angle. 

    Parameters
    ----------
    additional_observation : tuple, optional
        Tuple of np.ndarray. Contains ranges of additional features 
        for observation space. [0] should be low range, [1] should be 
        high range of additional observations. 

        The default observation space contains only the lidar 
        observation. 

        Example: 
        Child environment has additional goal position of x, y returned
        through _get_observation(). 
        
        low = np.array([-float('inf'), -float('inf')])
        high = np.array([float('inf'), float('inf')])
        super().__init__((low, high))
    '''
    metadata = {'render.modes': ['human']}
    
    def __init__(self, additional_observation=None):
        # Number of lidar segments for car 
        self.lidar_seg = 18

        # Set frameskip
        self.frame_skip = 1

        # Set up action space
        self.action_space = spaces.Box(np.array([0, 0, -.6]), 
            np.array([1, 1, .6]), dtype=np.float32)

        # Set up observation space with lidar 
        low = np.array([0.0 for i in range(self.lidar_seg)])
        high = np.array([1.0 for i in range(self.lidar_seg)])
        if additional_observation is not None: 
            low = np.concatenate((low, additional_observation[0]))
            high = np.concatenate((high, additional_observation[1]))
        self.observation_space = spaces.Box(low=low, high=high, 
            dtype=np.float32)
            
        self.reward_range = None
        
        # Connect client 
        self.client = p.connect(p.DIRECT)
        p.setTimeStep(1/120, self.client)
        self.closed = False
        # Random generator used for any randomly gen behavior
        self.random = np.random.RandomState()
        # Seed the random created above
        self.seed()

        # Set image for rendering
        self.imgsize = 100
        self.img = None
        
        # Used for reward function modification
        self.reward_func = None

    def modify(self, frame_skip=1, reward_func=None):
        ''' 
        Function used once after environment creation to specify frame 
        skip and a modified reward function. 

        See documentation for setting up a modified reward function.

        Parameters 
        ----------
        frame_skip : int, optional 
            Number of simulation time steps to apply the same action.
        reward_func : function, optional
            reward_func (bool, tuple of float); see documentation. 
        '''
        self.frame_skip= frame_skip
        self.reward_func = reward_func

    def step(self, action): 
        '''
        Applies an action and returns environment information.

        Parameters 
        ----------
        action : np.ndarray
            Action should be 1D array of size 3 with dtype np.float32.
            First index decides throttle, [0.0, 1.0]. 
            Second index decides break, [0.0, 1.0]. 
            Third index decides central steering angle, [-0.6, 0.6].

        Returns
        -------
        np.ndarray, float, bool, dict 
            Computed observation, reward, done state, and info after 
            the action is applied to the environment. 
        '''
        # Cast to np and clip to action space
        action = np.asarray(action)
        action = action.clip(self.action_space.low, self.action_space.high)
        # Perform action
        self._apply_action(action) 
        p.stepSimulation()
        self.timestep += 1
        for _ in range(1, self.frame_skip):
            if self._get_done(frame_skip=True):
                break
            self._apply_action(action) 
            p.stepSimulation()
            self.timestep += 1

        # Retrieve observation
        observation = self._get_observation()
        
        # Compute reward 
        reward = self._get_reward(observation)

        # Retrieve done status
        done = self._get_done()

        # Return observation, reward, done, {} 
        return observation, reward, done, dict()

    def reset(self):
        ''' 
        Initialization to start simulation. Loads all proper objects. 

        Can be overridden by inheriting classes to create specific 
        environment. Can be called from super().reset() to just reset 
        PyBullet client, gravity, plane, and car, timestep.
        '''
        # Default initialization of car, plane, and gravity 
        p.resetSimulation(self.client)
        p.setGravity(0,0,-10)
        self.plane = p.loadURDF(getResourcePath('plane/plane.urdf'), 
            physicsClientId=self.client)
        self.car = car.Car(self.lidar_seg, client=self.client)

        self.timestep = 0

        return self._get_observation()

    def render(self, mode='human'):
        '''
        Renders the environment through cars perspective. 

        Parameters
        ----------
        mode : str, optional
            Render mode according to metadata['render.modes']. 
        '''
        if self.img is None:
            self.img = plt.imshow(np.zeros((self.imgsize, self.imgsize, 4)))

        # Base information
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
        if self.img is not None: 
            plt.close()
        self.closed = True

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
        # Seed action space random sample generator
        self.action_space.seed(seeding.create_seed(seed, max_bytes=4))

        # Return seed used  
        return seed

    def _get_done(self): 
        ''' 
        Returns true if car has collision. Can be overridden by 
        other inheriting environments for specific done criteria. 

        Returns 
        -------
        bool 
            True if car encounters collision. 
        '''
        return self.car.get_collision()
    
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
        self.car.apply_action(action)

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
        return self.car.get_observation()

    def _get_reward(self, obs):
        ''' 
        Retrieves reward of car. None by default, must be overridden by
        inheriting classes to specify environment rewards. 

        Returns
        -------
        float
            Amount of reward at time step.
         '''
        raise NotImplementedError('Reward function not overridden.')

    def __del__(self): 
        ''' 
        Call close to remove from PyBullet client if user didn't close.
        '''
        if not self.closed: 
            self.close() 

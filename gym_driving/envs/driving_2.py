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

class Driving2(DrivingEnv):
    '''
    Drive towards a randomly placed target, with multiple obstacles in-
    between. Reaching the target is 20 reward, any collisions is -20.
    Moving away or towards the target at each step provides reward 
    equal to the difference in Euclidean distance from the previous 
    step. When the car is too close to the obstacles, reward 
    penalization inversely squared to the distance from the obstacles 
    is applied.

    Lidar is used to infer where obstacles are (see observation_space)
    where area around the car is divided into arcs of 20 degrees. If no
    obstacle is present in the arc, the arc's corresponding dimension is
    0. Closer obstacles will register with greater numbers, up to 1, 
    where an obstacle is directly in front of the car (collision). The
    middle, 9, of the 18 lidar observation dimensions corresponds to the
    20 degree arc directly in front of the car. 

    action_space : spaces.Box
        np.ndarray of size 3. [0, 1], [0, 1], [-.6, .6] range. First
        feature is throttle, second is break, third is central steering
        angle. 

    observation_space : spaces.Box
        np.ndarray of size 24. [0, 1] * 18, [-15, 15] * 2, [-1, 1] * 2,
        [-5, 5] * 2. First 18 are lidar observations. 19-20 is a vector
        to the target. 21-22 is the unit vector of car orientation. 
        23-24 is car's velocity vector.

    Maximum episode length of 1000. Frame skip and reward modification 
    easily available; see documentation. 
    '''
    def __init__(self):
        # Add car observations to observation space
        low = np.array([-15, -15, -1, -1, -5, -5])
        high = np.array([15, 15, 1, 1, 5, 5])
        super().__init__((low, high))

        self.done = False
        self.prev_dist = None
                                           
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
                        self.random.randint(2) else (second_coord, first_coord))

        # Default initialization of car, plane, and gravity 
        observation = super().reset()

        # Generate and place obstacles without overlap
        perpendicular = np.array((self.target[1], -self.target[0]))
        lower = self.target * 0.25
        upper = self.target * 0.8
        self.obstacles = []
        tries = 0
        while tries < 20:
            tries += 1
            ob = self.random.uniform(lower, upper) + (perpendicular * 
                self.random.uniform(-0.5, 0.5))
            size = self.random.randint(1, 5)
            for other_ob, other_size in self.obstacles: 
                if True in (abs(other_ob - ob) < (size / 10. + other_size/10.)):
                    break
            else: 
                self.obstacles.append((ob, size))
                Cube(list(ob) + [0], size, client=self.client)

        # Visual display of target
        Cube(list(self.target) +  [0], 2, marker=True, client=self.client)

        self.done = False
        self.prev_dist = np.linalg.norm(np.array(
            self.car.get_position_orientation()[0]) - self.target)

        return observation

    def _get_done(self, frame_skip=False): 
        ''' 
        Returns true if done.

        Parameters
        ----------
        frame_skip : bool, optional 
            If True, forces _get_done() to perform done calculations 
            without relying on other functions (like _get_reward()) to
            update the done status. 

        Returns 
        -------
        bool 
        '''
        if not frame_skip:
            return self.done

        currPos, _= self.car.get_position_orientation()
        # Terminal from episode length over 1000
        if self.timestep >= 1000: 
            return True
        # Terminal from driving off range
        if abs(currPos[0]) > 14.8 or abs(currPos[1]) > 14.8: 
            return True
        # Terminal from collision
        if self.car.get_collision(): 
            return True
        # Terminal from reaching target
        if np.linalg.norm(currPos - self.target) < 0.8: 
            return True
        return False

    def _get_observation(self): 
        ''' 
        Retrieves observation of car with lidar.

        Returns
        -------
        np.ndarray
            Lidar (18), vector to target (2), unit orientation (2), 
            velocity vector (2).
        '''
        lidar = self.car.get_lidar()
        pos, ori = self.car.get_position_orientation()
        vel = self.car.get_velocity()
        return np.concatenate((lidar, self.target - pos, ori, vel))

    def _get_reward(self, obs):
        ''' 
        Retrieves reward of car.

        Returns
        -------
        float
        Non-terminal: 
            A. Change from last step in Euclidean distance to target.
            B. List of distance to obstacles 
        Penalty for each individual obstacle is 0.01 / distance**2 if 
        distance < 1.6 else 0. 
        Returned float is A - sum(Penalty) over all obstacles. 

        Terminal: 
            A. 20 for reaching target, -20 for colliding with obstacle.
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

        # Terminal from collision
        if self.car.get_collision(): 
            self.done = True
            return (-20 if self.reward_func is None else
                self.reward_func(True, (-20,)))

        # Terminal from reaching target
        distance = np.linalg.norm(currPos - self.target)
        if distance < 0.8: 
            self.done = True
            return (20 if self.reward_func is None else
                self.reward_func(True, (20,)))

        # Change in distance
        delta_distance = (self.prev_dist - distance) 
        dist_to_obstacles = [np.linalg.norm(currPos - ob) - size / 10. for 
            ob, size in self.obstacles]
        # Scaled penalty for being too close to obstacles
        obstacle_penalty = 0        
        for dist in dist_to_obstacles: 
            obstacle_penalty += (0.01 / dist**2 if dist < 1.6 else 0)

        self.prev_dist = distance

        # Return either documented reward or scaled reward if there's a
        # reward function
        return (delta_distance - obstacle_penalty if self.reward_func is None
            else self.reward_func(False, (delta_distance, dist_to_obstacles)))

    def __del__(self): 
        ''' 
        Call super del for any additional cleanup. 
        '''
        super().__del__()

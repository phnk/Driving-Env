'''
Contains Car class.
'''
# External resources
import pybullet as p 
import numpy as np
# Local resources
from gym_driving.resources import _helper_functions as helper
from gym_driving.resources import getResourcePath

class Car: 
    '''
    Wrapper for a car URDF with interface for actions and observations.

    Parameters 
    ----------
    client : int, optional 
        PyBullet client to attach to, default of 0. 
    '''
    def __init__(self, client=0): 
        # Set client, load car 
        self.client = client 
        self.car = p.loadURDF(
            fileName=getResourcePath('racecar/racecar.urdf'),
            basePosition=[0, 0, 0],
            physicsClientId=self.client)

        # Joint indices for racecar/racecar.urdf
        joint_indices = { 
            'left_steering': 4, 
            'right_steering': 6, 
            'left_rear': 2, 
            'left_front': 5, 
            'right_rear': 3, 
            'right_front': 7
        }

        # Set joint arrays
        self.steer_wheels = [
            joint_indices['left_steering'],
            joint_indices['right_steering']
        ]
        self.drive_wheels = [
            joint_indices['left_rear'], 
            joint_indices['right_rear'],
            joint_indices['right_front'],
            joint_indices['left_front']
        ]
        self.num_drive_wheels = len(self.drive_wheels)
        self.num_steer_wheels = len(self.steer_wheels)

        self.wheel_velocity = 0

    def get_ids(self): 
        ''' 
        Returns tuple of (bodyid, clientid) of underlying car. 

        Returns 
        -------
        int, int 
            Body id, client id for underlying URDF. 
        '''
        return self.car, self.client

    def apply_action(self, action): 
        '''
        Takes throttle amount and steering position. 

        Parameters
        ----------
        action : float, float
            Throttle amount [0, 1], steering position [-0.5, 0.5].
        '''
        
        # Speed parameters
        throttle = action[0]
        max_torque = 30
        c_drag = 0.15
        c_rolling = 20
        c_break = -2
        car_mass = 5.81

        simulation_step = 0.004 
        wheel_radius = 0.05
        max_wheel_joint_speed = 100

        # Calculate speed from throttle 
        v = self.get_velocity()
        speed = np.linalg.norm(v) + 1e-9
        friction_force = v * -(c_drag * c_rolling * speed)
        wheel_force = self.get_orientation() * max_torque * throttle 
        self.wheel_velocity = self.wheel_velocity + \
            simulation_step * (wheel_force + friction_force) / car_mass
        wheel_joint_speed = np.linalg.norm(self.wheel_velocity) / wheel_radius
        # Limit speed based on constraints
        wheel_joint_speed = min(wheel_joint_speed, max_wheel_joint_speed)

        # Set steering position
        p.setJointMotorControlArray(
            self.car,
            self.steer_wheels,
            controlMode=p.POSITION_CONTROL, 
            targetPositions=[action[1]] * self.num_steer_wheels, 
            physicsClientId=self.client)
        # Set wheel velocity 
        p.setJointMotorControlArray(
            bodyUniqueId=self.car,
            jointIndices=self.drive_wheels,
            controlMode=p.VELOCITY_CONTROL, 
            targetVelocities=[wheel_joint_speed] * self.num_drive_wheels,
            physicsClientId=self.client)

    def get_observation(self, observation): 
        pass

    def get_speed(self): 
        ''' 
        Returns the speed of car in m/s. 

        Returns
        -------
        int 
            Speed of car in m/s. 
        '''
        return np.linalg.norm(self.get_velocity())

    def get_velocity(self): 
        '''
        Returns the velocity of the car in m/s for x, y.

        Returns
        -------
        np.ndarray 
            x, y, z speed in m/s.
        '''
        return np.array(p.getBaseVelocity(self.car, self.client)[0])[0:2]

    def get_orientation(self): 
        ''' 
        Returns unit orientation of car in x, y.
        '''
        angle = np.array(p.getEulerFromQuaternion(
            p.getBasePositionAndOrientation(self.car, self.client)[1])) 
        vec = (np.cos(angle[2]) * np.cos(angle[1]), 
               np.sin(angle[2]) * np.cos(angle[1]))
        vec = np.array(vec)
        return vec / np.linalg.norm(vec)


# External resources
import pybullet as p 
import pybullet_data
import numpy as np
# Local resources
from gym_driving.resources import _helper_functions as helper

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
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.car = p.loadURDF(
            fileName='racecar/racecar.urdf',
            basePosition=[0, 0, 0.1],
            physicsClientId=self.client)

        # Joint indicies for racecar/racecar.urdf
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
            joint_indices['right_rear']
        ]
        self.num_drive_wheels = len(self.drive_wheels)
        self.num_steer_wheels = len(self.steer_wheels)

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
        Takes wheel and steering torque and applies to car joints. 

        Parameters
        ----------
        action : list-like
            Variable argument iterable. 
        '''
        wheel_torque, steering_torque = action[:-1], action[-1]

        # Set steering position
        p.setJointMotorControlArray(
            self.car,
            self.steer_wheels,
            controlMode=p.VELOCITY_CONTROL, 
            targetPositions=[steering_torque] * self.num_steer_wheels, 
            physicsClientId=self.client)

        # Set wheel velocity 
        p.setJointMotorControlArray(
            bodyUniqueId=self.car,
            jointIndices=self.drive_wheels,
            controlMode=p.VELOCITY_CONTROL, 
            targetVelocities=wheel_torque,
            physicsClientId=self.client)

    def get_observation(self, observation): 
        pass



'''
Contains Car class.
'''
# External resources
import pybullet as p 
import numpy as np
import matplotlib.pyplot as plt
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
        self.joint_indices = { 
            'left_steering': 4, 
            'right_steering': 6, 
            'left_rear': 2, 
            'left_front': 5, 
            'right_rear': 3, 
            'right_front': 7
        }
        self.drive_wheels = [
            self.joint_indices['left_rear'], 
            self.joint_indices['right_rear'],
            self.joint_indices['right_front'],
            self.joint_indices['left_front']
        ]

        self.num_drive_wheels = len(self.drive_wheels)

        self.wheel_velocity = 0

        self.t = 0 
        self.vel = list()
        self.time = list()

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
            Throttle amount [0, 1], steering position [-0.6, 0.6].
        '''
        
        # Speed parameters
        throttle = action[0]
        max_torque = 100
        c_drag = 0.15
        c_rolling = 5
        c_break = -2
        car_mass = 4.73
        simulation_step = 0.004 
        wheel_radius = 0.05
        max_wheel_joint_speed = 120
        epsilon = 1e-15

        # Calculate speed from throttle 
        v = self.get_velocity()
        speed = np.linalg.norm(v) + epsilon
        friction_force = v * -(c_drag * speed)
        friction_force += (v * -c_rolling)
        wheel_force = self.get_orientation() * max_torque * throttle 
        acceleration = (wheel_force + friction_force) / car_mass
        new_velocity = v + simulation_step * acceleration
        wheel_joint_speed = np.linalg.norm(new_velocity) / wheel_radius
        # Limit speed based on constraints
        wheel_joint_speed = min(wheel_joint_speed, max_wheel_joint_speed)


        '''
        self.t += 1
        self.vel.append(np.linalg.norm(acceleration) )
        self.time.append(self.t) 
        if self.t > 5000: 
            plt.plot(self.time, self.vel)
            plt.show()
            exit()
        '''

        # Calculate steering angle from central steering
        central = action[1] + epsilon
        wheelbase = 0.325 
        half_width = 0.1
        left_wheel_angle = np.arctan(wheelbase / (wheelbase/np.tan(central) -
            half_width))
        right_wheel_angle = np.arctan(wheelbase / (wheelbase/np.tan(central) +
            half_width))

        # Set steering position
        p.setJointMotorControl2(self.car, self.joint_indices['left_steering'],
            controlMode=p.POSITION_CONTROL, targetPosition=left_wheel_angle,
            physicsClientId=self.client)
        p.setJointMotorControl2(self.car, self.joint_indices['right_steering'],
            controlMode=p.POSITION_CONTROL, targetPosition=right_wheel_angle,
            physicsClientId=self.client)
        # Set wheel velocity 
        p.setJointMotorControlArray(
            bodyUniqueId=self.car,
            jointIndices=self.drive_wheels,
            controlMode=p.VELOCITY_CONTROL, 
            targetVelocities=[wheel_joint_speed] * self.num_drive_wheels,
            forces=[1.2] * self.num_drive_wheels,
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


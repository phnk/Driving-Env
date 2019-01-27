'''
Contains Car class.
'''
# External resources
import pybullet as p 
import numpy as np
import math
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

        ids = p.loadURDF(
            fileName=getResourcePath('racecar/racecar.urdf'),
            basePosition=[1, 1, 0],
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
        self.joint_speed = 0

        # Max-min range of lidar
        self.lidar_range = 10
        self.min_lidar_range = 0.2
        # Angle covered by lidar 
        self.view_range = 180
        # Number of segments within covered area
        self.num_seg = 10
        
        # Determine ray start and end given above parameters
        self._init_lidar()

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
        Takes throttle amount, breaking amount, and steering position. 

        If any breaking is specified, throttle is not applied. 

        Parameters
        ----------
        action : float, float, float
            Throttle [0, 1], break [0, 1], steering position [-0.6, 0.6].
        '''
        
        # Speed parameters
        throttle = action[0]
        breaking = action[1]
        max_torque = 120
        c_drag = 0.02
        c_rolling = 0.5
        c_break = -50
        simulation_step = 0.004 

        # Calculate speed with friction 
        force = c_break * breaking if breaking else max_torque * throttle
        friction = -self.joint_speed * (self.joint_speed * c_drag + c_rolling)
        acceleration = force + friction
        self.joint_speed = self.joint_speed + simulation_step * acceleration
        if self.joint_speed < 0: 
            self.joint_speed = 0
        
        # Calculate steering angle from central steering
        central = action[2] 
        wheelbase = 0.325 
        half_width = 0.1
        if central == 0: 
            left_wheel_angle = right_wheel_angle = 0
        else: 
            left_wheel_angle = np.arctan(wheelbase / (wheelbase/np.tan(central) 
                - half_width))
            right_wheel_angle = np.arctan(wheelbase / (wheelbase/np.tan(central)
                + half_width))

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
            targetVelocities=[self.joint_speed] * self.num_drive_wheels,
            forces=[1.2] * self.num_drive_wheels,
            physicsClientId=self.client)

    def get_observation(self): 
        '''
        Returns an observation.

        Returns 
        -------
        To be decided. 

        Current: 
        position(2), orientation(2), velocity(2), wheel angle 
        [-inf, inf], [-1, 1], [-5, 5], [-.7, .7]
        '''
        pos, ori = self.get_position_orientation() 
        vel = self.get_velocity()
        avg_wheel_angle = np.array((p.getJointState(self.car,
            self.joint_indices['left_steering'], self.client)[0] + 
            p.getJointState(self.car, self.joint_indices['right_steering'], 
            self.client)[0]) / 2).reshape(-1)
        return np.concatenate((pos, ori, vel, avg_wheel_angle))

        
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

    def get_position_orientation(self): 
        ''' 
        Returns position and unit orientation of car in x, y.

        Returns 
        -------
        np.ndarray, np.ndarray
        Position and unit orientation of car in x, y coordinates.  
        '''
        pos_ori = p.getBasePositionAndOrientation(self.car, self.client)
        angle = p.getEulerFromQuaternion(pos_ori[1])
        ori = np.array([np.cos(angle[2]) * np.cos(angle[1]), 
                        np.sin(angle[2]) * np.cos(angle[1])])
        pos = np.array(pos_ori[0][:2])
        return pos, ori

    def get_lidar(self): 
        '''
        Returns lidar observation.
        '''
        batch = p.rayTestBatch(self.start_rays, self.end_rays, self.car, 
            9, self.client)
        for laser in batch: 
            print(laser[0], end=' ')
        print()
        lidar_ob = np.zeros(self.num_seg)
        ray_per_zone = self.num_rays // self.num_seg
        for zone in range(self.num_seg): 
            for ray in range(ray_per_zone):
                if batch[zone * ray_per_zone + ray][0] != -1: 
                    score = 1 - batch[zone * ray_per_zone + ray][2] 
                    lidar_ob[zone] = max(lidar_ob[zone], score)
        # print([round(i, 1) for i in lidar_ob])

    def _init_lidar(self): 
        ''' 
        Performs computations to set up lidar beams. 
        '''
        # Number of rays in lidar
        self.num_rays = math.ceil(int(self.view_range / 5) / self.num_seg) * \
            self.num_seg
        # Initialize start and end of rays given params
        starting_degree = -(self.view_range - self.view_range / 2) + 90
        change_degree = self.view_range / self.num_rays
        self.start_rays = [
            [self.min_lidar_range
             * math.sin(math.radians(starting_degree + i * change_degree)),
             self.min_lidar_range
             * math.cos(math.radians(starting_degree + i * change_degree)),
             0.05] for i in range(self.num_rays)]
        self.end_rays = [
            [self.lidar_range * 
             math.sin(math.radians(starting_degree + i * change_degree)),
             self.lidar_range * 
             math.cos(math.radians(starting_degree + i * change_degree)),
             0.05] for i in range(self.num_rays)]

    def _debug_lidar(self): 
        '''
        Displays lidar sweeps in p.GUI client.
        '''
        for s, e in zip(self.start_rays, self.end_rays):
            p.addUserDebugLine(s, e, [1, 0, 0], parentObjectUniqueId=self.car, 
                parentLinkIndex=9, lifeTime=0.3)


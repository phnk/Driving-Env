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
    def __init__(self, lidar_seg, client=0): 
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
        self.joint_speed = 0

        # Max-min range of lidar
        self.lidar_range = 15
        self.min_lidar_range = 0.2
        # Number of segments within covered area
        self.num_seg = lidar_seg
        
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
        max_torque = 150
        c_drag = 0.01
        c_rolling = 0.3
        c_break = -125
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
        pos, ori, angle = self.get_position_orientation(True)
        lidar = self.get_lidar(angle)
        vel = self.get_velocity()
        avg_wheel_angle = np.array((p.getJointState(self.car,
            self.joint_indices['left_steering'], self.client)[0] + 
            p.getJointState(self.car, self.joint_indices['right_steering'], 
            self.client)[0]) / 2).reshape(-1)
        return np.concatenate((pos, ori, vel, avg_wheel_angle, lidar))
        
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

    def get_collision(self): 
        ''' 
        Returns true if collided, false if no collision. 

        Returns
        -------
        bool 
        Status of collision of car. 
        '''
        contact = p.getContactPoints(bodyA=self.car, linkIndexA=0,
            physicsClientId=self.client)
        return True if len(contact) else False

    def get_position_orientation(self, angle=False): 
        ''' 
        Returns position and unit orientation in x, y coordinates.

        Parameters 
        ----------
        angle : boolean, optional
            If true, also returns the angle of the car in radians.

        Returns 
        -------
            angle : False
        np.ndarray, np.ndarray
        Position, orientation car in x, y coordinates

            angle : True
        np.ndarray, np.ndarray, float
        Position, orientation car in x, y coordinates, angle of car.
        Angle is between [-pi, pi]. 
        '''
        pos, ang = p.getBasePositionAndOrientation(self.car, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = np.array([math.cos(ang[2]), math.sin(ang[2])])
        return (pos[:2], ori, ang[2]) if angle else (pos[:2], ori)

    def get_lidar(self, angle=None): 
        '''
        Returns lidar observation.

        Breaks region around car into number of segments specified in 
        init as self.num_seg, returning an array of self.num_seg 
        length. The value in each region is a score of [0, 1], with 
        regions containing objects closer to the car having a score 
        closer to 1. 

        E.G. Region [0] has an object very close, it's score is 0.94. 
        Region [1] has an object near the edge of lidar detection, its
        score is 0.01. 

        Regions that are clear of objects will have a score of 0. 

        Parameters
        ----------
        angle : int, optional 
            Optionally pass angle of car in radians to prevent overhead
            of additional call to get angle of car. 
        
        Returns
        -------
        np.ndarray
            The scores for each region as described above.
        '''
        if angle is None: 
            angle = p.getEulerFromQuaternion(
                p.getBasePositionAndOrientation(self.car, self.client)[1])[2]
        sensor_info = p.getLinkState(self.car, 8, computeForwardKinematics=True,
            physicsClientId=self.client)
        pos = np.array(sensor_info[0])
        batch = p.rayTestBatch(self.start_rays + pos, self.end_rays + pos, 
            self.car, 9, self.client)
        lidar_ob = np.zeros(self.num_seg)
        ray_per_zone = self.num_rays // self.num_seg
        offset = -int((angle / math.pi) * self.num_rays / 2)
        for zone in range(self.num_seg): 
            for ray in range(ray_per_zone):
                index = ((zone * ray_per_zone + ray) + offset)
                if index >= len(batch): 
                    index %= len(batch)
                if batch[index][0] != -1: 
                    score = 1 - batch[index][2] 
                    lidar_ob[zone] = max(lidar_ob[zone], score)
        return lidar_ob

    def _init_lidar(self): 
        ''' 
        Performs computations to set up lidar beams. 
        '''
        # Number of rays in lidar
        self.num_rays = (math.ceil(24 * self.lidar_range) //
            self.num_seg * self.num_seg)
        # Initialize start and end of rays given params
        starting_degree = -90
        change_degree = 360 / self.num_rays
        self.start_rays = [
            np.array([self.min_lidar_range
             * math.sin(math.radians(starting_degree + i * change_degree)),
             self.min_lidar_range
             * math.cos(math.radians(starting_degree + i * change_degree)),
             0.0]) for i in range(self.num_rays)]
        self.end_rays = [
            np.array([self.lidar_range * 
             math.sin(math.radians(starting_degree + i * change_degree)),
             self.lidar_range * 
             math.cos(math.radians(starting_degree + i * change_degree)),
             0.0]) for i in range(self.num_rays)]
        self.start_rays = np.array(self.start_rays)
        self.end_rays = np.array(self.end_rays)

    def _debug_lidar(self): 
        '''
        Displays lidar sweeps in p.GUI client.
        '''
        for s, e in zip(self.start_rays, self.end_rays):
            p.addUserDebugLine(s, e, [1, 0, 0], parentObjectUniqueId=self.car, 
                parentLinkIndex=8, lifeTime=0.5)

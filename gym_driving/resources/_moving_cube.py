''' 
Contains moving cube class as URDF wrapper. 
'''
import pybullet as p 
import numpy as np
from gym_driving.resources import getResourcePath
from gym_driving.resources._cube import Cube

class MovingCube(Cube): 
    '''
    Moving cube class. 

    Parameters
    ----------
    position : list
        Starting position of cube. Default [0, 0, 0].
    size : int, optional
        Size of 1, 2, 3, 4, or 5, larger being a larger cube. Default 1.
    client : int, optional 
        Physics client, default of 0. 
    end_position : list, optional 
        Where to move cube. Default [1, 0, 0]
    '''
    def __init__(self, position, end_position, size=1, client=0):
        super().__init__(position, size, client=client)
        position = np.array(position)
        self.end_position = np.array(end_position)

        self.speed_norm = 1 / np.linalg.norm(self.end_position - position)
        self.midpoint = ((self.end_position - position) / 2 + position)

    def step(self): 
        '''
        Performs step to move between end and starting position.

        Returns 
        -------
            x, y of cubes position.   
        '''
        pos, ori = p.getBasePositionAndOrientation(self.cube, self.client)
        pos = np.array(pos) 

        p.applyExternalForce(self.cube, -1, (self.midpoint - pos) * 
            self.speed_norm, pos, p.WORLD_FRAME, physicsClientId=self.client)

        self.position = pos[:2]

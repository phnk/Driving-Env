''' 
Contains basic cube class as URDF wrapper. 
'''
import pybullet as p 
import numpy as np
from gym_driving.resources import getResourcePath

class Cube: 
    '''
    Basic cube class. 

    Parameters
    ----------
    position : list
        Three floats of position in x, y, z coordinates for cube. 
    size : int, optional
        Size of 1, 2, 3, 4, or 5, larger being a larger cube. Default 1.
    marker : bool, optional 
        Load marker cube; doesn't show on lidar. 
    client : int, optional 
        Physics client, default of 0. 
    '''
    # Supported sizes
    sizes = {1, 2, 3, 4, 5}

    def __init__(self, position, size=1, marker=False, client=0):
        assert size in Cube.sizes, f'Unsupported size {size} for Cube.'

        self.client = client
        self.position = np.array(position[0:2])
        if not marker: 
            self.cube = p.loadURDF(
                fileName=getResourcePath(f'cube/cube{size}.urdf'), 
                basePosition=position, 
                physicsClientId=self.client)
        else: 
            self.cube = p.loadURDF(
                fileName=getResourcePath('cube/marker.urdf'), 
                basePosition=position, 
                physicsClientId=self.client)

    def get_ids(self): 
        ''' 
        Returns tuple of (bodyid, clientid) of underlying cube. 

        Returns 
        -------
        int, int 
            Body id, client id for underlying URDF. 
        '''
        return self.cube, self.client
   
    def get_position(self): 
        '''
        Returns tuple of position. 

        Returns 
        -------
        np.ndarray
            x, y of cubes position.   
        '''
        return self.position

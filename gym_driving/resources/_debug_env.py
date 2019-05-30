'''
Debug environment for testing new objects or functionality with GUI. 
'''
import numpy as np
import pybullet as p
from gym_driving.resources import getResourcePath
import gym_driving.resources._helper_functions as helper
import gym_driving.resources._car as car
from gym_driving.resources._moving_cube import MovingCube
from gym_driving.resources._cube import Cube
import time

def main(): 
    p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.setTimeStep(1/120)
    t = p.addUserDebugParameter('Throttle', 0, 1, 0)
    b = p.addUserDebugParameter('Break Position', 0, 1, 0)
    s = p.addUserDebugParameter('Steering Position', -.6, .6, 0)
    f = p.addUserDebugParameter('Force', -10, 10, 0)
    quit = p.addUserDebugParameter('Quit', 0, .01, 0)
    c = car.Car(10)
    plane = p.loadURDF(getResourcePath('plane/plane.urdf'))

    while True: 
        throttle = p.readUserDebugParameter(t)
        breaking = p.readUserDebugParameter(b)
        steer = p.readUserDebugParameter(s)
        quitting = p.readUserDebugParameter(quit)
        if quitting: 
            p.disconnect(0)
            exit()
        c.apply_action([throttle, breaking, steer])
        p.stepSimulation()
        time.sleep(0.003) 

if __name__ == '__main__': 
    main()

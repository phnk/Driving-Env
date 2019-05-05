'''
Debug environment for testing new objects or functionality with GUI. 
'''
import numpy as np
import pybullet as p
from gym_driving.resources import getResourcePath
import gym_driving.resources._helper_functions as helper
import gym_driving.resources._car as car
from gym_driving.resources._moving_cube import MovingCube
import time

def main(): 
    p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    t = p.addUserDebugParameter('Throttle', 0, 1, 0)
    b = p.addUserDebugParameter('Break Position', 0, 1, 0)
    s = p.addUserDebugParameter('Steering Position', -.6, .6, 0)
    f = p.addUserDebugParameter('Force', -10, 10, 0)
    quit = p.addUserDebugParameter('Quit', 0, .01, 0)
    c = car.Car(10)
    a = p.loadURDF(getResourcePath('plane/plane.urdf'))

    cub = MovingCube([3, 0, 0], [1, 5, 0], size=3)

    count = 0
    while True: 
        count += 1
        throttle = p.readUserDebugParameter(t)
        breaking = p.readUserDebugParameter(b)
        steer = p.readUserDebugParameter(s)
        quitting = p.readUserDebugParameter(quit)
        if quitting: 
            p.disconnect(0)
            exit()
        c.apply_action([throttle, breaking, steer])
        cub.step()
#        print(cub.get_position())
        p.stepSimulation()
        time.sleep(0.001) 
        if count > 1200: 
            exit()

if __name__ == '__main__': 
    main()

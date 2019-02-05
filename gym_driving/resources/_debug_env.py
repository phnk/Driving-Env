'''
Debug environment for testing new objects or functionality with GUI. 
'''
import numpy as np
import pybullet as p
from gym_driving.resources import getResourcePath
import gym_driving.resources._helper_functions as helper
import gym_driving.resources._car as car
import gym_driving.resources._cube as cube
import time

def main(): 
    p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    t = p.addUserDebugParameter('Throttle', 0, 1, 0)
    b = p.addUserDebugParameter('Break Position', 0, 1, 0)
    s = p.addUserDebugParameter('Steering Position', -.6, .6, 0)
    quit = p.addUserDebugParameter('Quit', 0, .01, 0)
    c = car.Car(10)
    a = p.loadURDF(getResourcePath('plane/plane.urdf'))

    cube.Cube([3, 0, 0], size=5)
    cube.Cube([-1, 3, 0], size=5)
    count = 0
    while True: 
        count += 1
        throttle = p.readUserDebugParameter(t)
        breaking = p.readUserDebugParameter(b)
        steer = p.readUserDebugParameter(s)
        quitting = p.readUserDebugParameter(quit)
        if quitting: 
            p.disconnect()
            quit()
        c.apply_action([throttle, breaking, steer])
        p.stepSimulation()
        col = c.get_collision()
        if col: 
            print('collision')
        #if count % 100 == 0: 
        #    c._debug_lidar()
        #    print(c.get_lidar())
        time.sleep(0.001) 

if __name__ == '__main__': 
    main()

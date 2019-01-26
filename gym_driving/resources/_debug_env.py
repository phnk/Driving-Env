import numpy as np
import pybullet as p
from gym_driving.resources import getResourcePath
import gym_driving.resources._helper_functions as helper
import gym_driving.resources._car as car
import time



def main(): 
    p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    t = p.addUserDebugParameter('Throttle', 0, 1, 0)
    b = p.addUserDebugParameter('Break Position', 0, 1, 0)
    s = p.addUserDebugParameter('Steering Position', -.6, .6, 0)
    c = car.Car()
    p.loadURDF(getResourcePath('plane/plane.urdf'))
    while True: 
        throttle = p.readUserDebugParameter(t)
        breaking = p.readUserDebugParameter(b)
        steer = p.readUserDebugParameter(s)
        c.apply_action([throttle, breaking, steer])
        p.stepSimulation()
        time.sleep(0.004) 

    p.disconnect()


if __name__ == '__main__': 
    main()

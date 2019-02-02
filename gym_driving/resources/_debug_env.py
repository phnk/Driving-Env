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
    quit = p.addUserDebugParameter('Quit', 0, .01, 0)
    c = car.Car(10)
    a = p.loadURDF(getResourcePath('plane/plane.urdf'))
    p.loadURDF(getResourcePath('racecar/racecar.urdf'), basePosition=[2, 2, 0])
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
        if count % 100 == 0: 
            ob = c.get_lidar()
            print([round(i, 1) for i in ob])
        time.sleep(0.001) 



if __name__ == '__main__': 
    main()

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

prev_dist = None

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
    a = p.loadURDF(getResourcePath('plane/plane.urdf'))


    target, obstacles = reset()
    global prev_dist
    prev_dist = np.linalg.norm(np.array(
        c.get_position_orientation()[0]) - target)


    while True: 
        print(_get_reward(c, target, obstacles))
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


def reset():
    ''' 
    Initialization to start simulation. Loads all proper objects. 
    '''
    # Generate new target each episode
    first_coord = np.random.randint(-13, 13)
    second_coord = np.random.randint(0, 13) if abs(first_coord) > 5 else \
                    np.random.randint(5, 13)
    second_coord *= np.random.choice([-1, 1])
    target = np.array((first_coord, second_coord) if
                    np.random.randint(2) else (second_coord, first_coord))
    target = np.array((2, 10))

    # Generate and place obstacles without overlap
    perpendicular = np.array((target[1], -target[0]))
    lower = target * 0.25
    upper = target * 0.8
    obstacles = []
    tries = 0
    while tries < 20:
        tries += 1
        ob = np.random.uniform(lower, upper) + (perpendicular * 
            np.random.uniform(-0.5, 0.5))
        size = np.random.randint(1, 5)
        for other_ob, other_size in obstacles: 
            if True in (abs(other_ob - ob) < (size / 10. + other_size/10.)):
                break
        else: 
            size = np.random.randint(1, 5)
            obstacles.append((ob, size))
            Cube(list(ob) + [0], size)

    # Visual display of target
    Cube(list(target) +  [0], 2, marker=True)
    global prev_dist


    return target, obstacles 

def _get_reward(car, target, obstacles):
    ''' 
    Retrieves reward of car.

    Returns
    -------
    float
    Non-terminal: 
        A. Change from last step in Euclidean distance to target.
        B. List of distance to obstacles 
    Penalty for each individual obstacle is 0.01 / distance**2 if 
    distance < 1.6 else 0. 
    Returned float is A - sum(Penalty) over all obstacles. 

    Terminal: 
        A. 20 for reaching target, -20 for colliding with obstacle.
    Returned float is value above directly.
    '''
    global prev_dist

    currPos, _= car.get_position_orientation()

    # Terminal from driving off range
    if abs(currPos[0]) > 14.8 or abs(currPos[1]) > 14.8:
        return 0

    # Terminal from collision
    if car.get_collision(): 
        return -20

    # Terminal from reaching target
    distance = np.linalg.norm(currPos - target)
    if distance < 0.8: 
        return 20

    # Change in distance
    delta_distance = (prev_dist - distance) 
    dist_to_obstacles = [np.linalg.norm(currPos - ob) - size / 10. for 
        ob, size in obstacles]
    # Scaled penalty for being too close to obstacles
    obstacle_penalty = 0        
    for dist in dist_to_obstacles: 
        obstacle_penalty += (0.01 / dist**2 if dist < 1.6 else 0)

    prev_dist = distance

    # Return either documented reward or scaled reward if there's a
    # reward function
    return delta_distance - obstacle_penalty 


if __name__ == '__main__': 
    main()




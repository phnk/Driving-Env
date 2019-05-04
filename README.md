# gym-driving
Set of OpenAI gym environments built on PyBullet; in development. 

### To Install: 
Clone Repo: 
```bash 
git clone https://github.com/GerardMaggiolino/Driving-Env.git
```
Pip Install: 
```bash
cd Driving-Env
pip install . 
```

### To Run: 
```python
import gym 
import gym_driving

env = gym.make('Driving-v0')
```
and interact with the environment as you would for any OpenAI gym environment! 

## Current Environments and Special Features 

Add frame skip and modify default reward functions with the **modify** function. 
```python
env.modify(frame_skip=4, reward_func=my_reward_function)
```
Documentation on reward_func *coming soon.* 

### Driving-v0

[<img src="https://github.com/GerardMaggiolino/Driving-Env/blob/master/demo/Driving-v0-Screenshot.png" height=300 width=300>](https://github.com/GerardMaggiolino/Driving-Env/blob/master/demo/Driving-v0-Example.mov)
  
**Drive towards a randomly placed target.** Reaching the target yields
50 reward. Moving away or towards the target at each step provides
reward equal to the difference in Euclidean distance from the 
previous step.

The action_space is spaces.Box of size 3. The ranges are [0, 1], [0, 1], [-.6, .6]. The first feature is throttle, second is break, and third is central steering angle. 

The observation_space is spaces.Box of size 6. The ranges are [-15, 15] * 2, [-1, 1] * 2, [-5, 5] * 2. The first pair is a vector to the target, the second is the unit vector of car orientation, and the third is a velocity vector.

### Driving-v1 

[<img src="https://github.com/GerardMaggiolino/Driving-Env/blob/master/demo/Driving-v1-Screenshot.png" height=300 width=300>](https://github.com/GerardMaggiolino/Driving-Env/blob/master/demo/Driving-v1-Example.mov)
  
**Drive towards a randomly placed target, with an obstacle in-between**.
Reaching the target yields 50 reward, colliding with the obstacle is
-50. Moving away or towards the target at each step provides
reward equal to the difference in Euclidean distance from the 
previous step. When the car is too close to the obstacle, reward
penalization inversely squared to the distance from the obstacle is
applied. 

Lidar is used to infer where the obstacle is (see observation_space)
where area around the car is divided into arcs of 20 degrees. If no
obstacle is present in the arc, the arc's corresponding dimension is
0. Closer obstacles will register with greater numbers, up to 1. The
9th observation corresponds to the 20 degree arc directly in front of 
the car. 

The action_space is spaces.Box of size 3. The ranges are [0, 1], [0, 1], [-.6, .6]. The first feature is throttle, second is break, and third is central steering angle. 

The observation_space is spaces.Box of size 24. The ranges are [0, 1] * 18, [-15, 15] * 2, [-1, 1] * 2, [-5, 5] * 2. The first 18 are lidar observations. 19-20 is a vector to the target. 21-22 is the unit vector of car orientation. 23-24 is car's velocity vector.

### Driving-v2 

*Coming soon.*

### Driving-v3

*Coming soon.*

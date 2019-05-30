# gym_driving
Set of OpenAI gym environments built on PyBullet; in development. 

#### To Install: 
Clone Repo: 
```bash 
git clone https://github.com/GerardMaggiolino/Driving-Env.git
```
Pip Install: 
```bash
cd Driving-Env
pip install . 
```

#### To Run: 
```python
import gym 
import gym_driving

env = gym.make('Driving-v0')
```
and interact with the environment as you would for any OpenAI gym environment! 

# Current Environments and Special Features 

Add frame skip and modify default reward functions with the **modify** function. 
```python
env.modify(frame_skip=4, reward_func=my_reward_function)
```
Documentation on reward_func *coming soon.* 

### Driving-v0

[<img src="https://github.com/GerardMaggiolino/Driving-Env/blob/master/demo/Driving-v0-Screenshot.png" height=300 width=300>](https://github.com/GerardMaggiolino/Driving-Env/blob/master/demo/Driving-v0-Example.mov)
  
**Drive towards a randomly placed target.** 

Reaching the target yields
20 reward. Moving away or towards the target at each step provides
reward equal to the difference in Euclidean distance from the 
previous step.

**action_space: spaces.Box(3)** 

| Dimension | Range | Feature Description | 
| :---: | :---: | :---: |
| 0 | [0, 1] | Throttle | 
| 1 | [0, 1] | Break | 
| 2 | [-0.6, 0.6] | Steering Angle | 

**observation_space: spaces.Box(6)** 

| Dimension | Range | Feature Description | 
| :---: | :---: | :---: |
| 0 - 1 | [-15, 15] | Vector to Target | 
| 2 - 3 | [-1, 1] | Unit Car Orientation | 
| 4 - 5 | [-5, 5] | Velocity Vector | 

### Driving-v1 

[<img src="https://github.com/GerardMaggiolino/Driving-Env/blob/master/demo/Driving-v1-Screenshot.png" height=300 width=300>](https://github.com/GerardMaggiolino/Driving-Env/blob/master/demo/Driving-v1-Example.mov)
  
**Drive towards a randomly placed target, with an obstacle in-between**.
Reaching the target yields 20 reward, colliding with the obstacle is
-20. Moving away or towards the target at each step provides
reward equal to the difference in Euclidean distance from the 
previous step. When the car is too close to the obstacle, reward
penalization inversely squared to the distance from the obstacle is
applied. 

Lidar is used to infer where the obstacle is
where area around the car is divided into arcs of 20 degrees. If no
obstacle is present in the arc, the arc's corresponding dimension has a value of
0. Closer obstacles will register with greater numbers, up to 1. 

**action_space: spaces.Box(3)** 

*Same as Driving-v0.*

**observation_space: spaces.Box(24)** 

| Dimension | Range | Feature Description | 
| :---: | :---: | :---: |
| 0 - 17 | [0, 1] | Lidar | 
| 18 - 19 | [-15, 15] | Vector to Target | 
| 20 - 21 | [-1, 1] | Unit Car Orientation | 
| 22 - 23 | [-5, 5] | Velocity Vector | 

### Driving-v2 

[<img src="https://github.com/GerardMaggiolino/Driving-Env/blob/master/demo/Driving-v2-Screenshot.png" height=300 width=300>](https://github.com/GerardMaggiolino/Driving-Env/blob/master/demo/Driving-v2-Example.mov)

**Drive towards a randomly placed target, with multiple obstacles in-between.**

Same reward as Driving-v1, with multiple obstacles of different sizes. Penalty for distance to obstacles is a summation of the same function of Driving-v1 over each obstacle. Obstacle's positions and sizes are randomized, but between the car and target.

**action_space: spaces.Box(3)** 

*Same as Driving-v0.*

**observation_space: spaces.Box(24)** 

*Same as Driving-v1.*

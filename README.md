# gym-driving
Eventually an OpenAI gym environment. 

### To install: 

When in gym-driving directory, 
```bash
pip install -e . 
```

### To run: 

```python
import gym 
import gym_driving

env = gym.make('Driving-v0')
env = gym.make('DrivingContinuous-v0')
```
Documentation on action spaces (defined through gym.spaces interface) in 
gym_driving/envs/driving_discrete.py and gym_driving/envs/driving_continuous.py. 

### To edit: 

Edit class DrivingEnv in gym_driving/envs/driving_env.py

Add any additional resources gym_driving/resources/.

### Development Notes: 
Please add any local URDFs under gym_driving/resources. Use getResourcePath 
(this is already imported in driving_env) with the filename when loading the
URDF to get a system independent path. 

Recommendation - for objects being created in the environment, such as the
blocks, created separate Python object wrappers. This will make the environments
much easier to control, maintain, and extend upon. E.G.

```python 
class BlockObj: 
    def __init__(self, placement): 
        p.loadURDF('block.urdf')
        # More object init stuff  

    # More object specific transformations and observations 

...
...

# Manipulate objects through an interface like this in DrivingEnv
b1 = BlockObj([5, -3, 10])
b1.getDistance(self.car)
```

### TODO: 

#### Movable Block Objects (PRIORITY):
1. Create and position URDF objects. 
  - Position objects relative to other objects. 
  - Create URDFs with collision properties. 
2. Observation 
  - Find angle and distance between car model and other created URDF objects.

#### Environment Generation: 
1. <s>Have camera lock and follow car.</s> This is done! 
2. Have objects be generated around car, rather than rigidly placed. 

  
### FINISHED 

#### Realistic Car Model (PRIORITY): 

1. Find open source or build a more realistic URDF car model. 
  -  <i>This should have Ackerman steering and friction between components. Maybe even a driveshaft and steering column!</i>
2. a Action
  - Continuous application of torque to steering column adjust wheel angle. 
  - Continuous application of torque to driveshaft to adjust car velocity.   
2. b Observation
  - Obtain car speed and wheel positions or angle from model.

#### Environment Optimization: 
1. Set PyBullet client of p.DIRECT. Render using TinyRenderer rather than OpenGL.
  - This means that we'll only render when the render() function is called. 

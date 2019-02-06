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
```

### To edit: 

Edit class DrivingEnv in gym_driving/envs/driving_env.py

Add any additional resources gym_driving/resources/.

Please add any local URDFs under gym_driving/resources. Use getResourcePath 
(this is already imported in driving_env) with the filename when loading the
URDF to get a system independent path. 

### TODO: 

#### Task based environments: 
1. Environment with a reward function and concrete task. 
  - Inherit from DrivingEnv and override the _get_observation() and reset() methods. 
  - Build the environment in reset(). 
  - Return specific observations in _get_observation() (EG the location of a goal). 
2. Any environment you can think of! 

  
### FINISHED: 

#### Movable Block Objects (PRIORITY):
1. Create and position URDF objects. 
  - Position objects relative to other objects. 
  - Create URDFs with collision properties. 
2. Observation 
  - Find angle and distance between car model and other created URDF objects.

#### Render of Car: 
1. Have camera lock and follow car. 

#### Lidar Observation (PRIORITY): 

1. Create a lidar based collision observation model using ray casts. 
  - Segmented observations in arc around car. 
  - Sense objects with detectable collision zones. 
  - Update car and environment observations. 

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

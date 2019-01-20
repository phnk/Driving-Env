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

env = gym.make('driving-v0')
```

### To edit: 

Edit class DrivingEnv in gym_driving/envs/driving_env.py

Add any additional resources gym_driving/resources/.

### Development Notes: 
Please add any local URDFs under gym_driving/resources. Use getResourcePath 
(this is already imported in driving_env) with the filename when loading the
URDF to get a system independent path. 

### TODO: 

#### Realistic Car Model: 
1. Find open source or build a more realistic URDF car model. 
  -  <i>This should have Ackerman steering and friction between components. Maybe even a driveshaft and steering column!</i>
2. a Action 
  - Continuous application of torque to steering column adjust wheel angle. 
  - Continuous application of torque to driveshaft to adjust car velocity.   
2. b Observation 
  - Obtain car speed and wheel positions or angle from model. 

#### Movable Block Objects: 
1. Create and position URDF objects. 
  - Position objects relative to other objects. 
  - Create URDFs with collision properties. 
2. Observation  
  - Find angle and distance between car model and other created URDF objects.


    

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

Add any additional resources gym_driving/resources/ 

### Development Notes: 
Import and use getResourcePath in gym_driving/resources to get the absolute, 
platform independent path name to any resources in the resources directory, such
as URDF files.

#### TODO: 
Develop minimal environment: 
    - Car controllable through actions with env.step(<i>action</i>), with
    observation returned. 
    - Create controllable car. 
    - Create observation of space. 


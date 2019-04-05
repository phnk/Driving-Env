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

Create documentation. Create new environments, test existing built one (Driving-v0) with baselines algorithms. 

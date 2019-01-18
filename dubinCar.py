import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
parentdir = os.path.join(currentdir,"../gym")

os.sys.path.insert(0,parentdir)

import pybullet as p
import pybullet_data
import keyboard as k
import dubins

import time

#define variables
RIGHT = -1
LEFT = 1
TYPE_OF_PATH = ["LSL", "LSR", "RSL", "RSR", "RLR", "LRL" ]

def update_steering ( steeringAngle ) :
	print('steering: ', steeringAngle)
	for steer in steering:
		p.setJointMotorControl2(car,steer,p.POSITION_CONTROL,targetPosition=steeringAngle)
	time.sleep(2)
	
def update_velocity ( velocity ) :
	for wheel in inactive_wheels:
		p.setJointMotorControl2(car,wheel,p.VELOCITY_CONTROL,targetVelocity=velocity,force=10)

def CSC ( curve1, curve2 ) :
	#start moving
	update_velocity( 10 )
	#turn first curve
	update_steering( curve1 * 0.8 )
	#turn forward
	update_steering( 0 )
	#turn second curve
	update_steering( curve2 * 0.8 )
	#stop moving
	update_velocity( 0 )

def CCC ( curve1, curve2, curve3 ) :
	update_velocity( 10 )
	#turn first curve
	update_steering( curve1 * 0.8 )
	#turn forward
	update_steering( curve2 * 0.8 )
	#turn second curve
	update_steering( curve3 * 0.8 )
	#stop moving
	update_velocity( 0 )



q0 = (0, 0, 0)
q1 = (100, 80, 0)
turning_radius = 1.0
path = dubins.shortest_path(q0, q1, turning_radius)
print('')
print( 'shortest path from', q0, ' to ', q1 )
print( 'length of shortest path:',  path.path_length() )
print( 'type of path is: ', TYPE_OF_PATH[path.path_type()] )
print('')



cid = p.connect(p.SHARED_MEMORY)
if (cid<0):
	p.connect(p.GUI)
	
p.resetSimulation()
p.setGravity(0,0,-10)

useRealTimeSim = 1

#for video recording (works best on Mac and Linux, not well on Windows)
#p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "racecar.mp4")
p.setRealTimeSimulation(useRealTimeSim) # either this
#p.loadURDF("plane.urdf")
p.loadSDF(os.path.join(pybullet_data.getDataPath(),"stadium.sdf"))

car = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"racecar/racecar.urdf"))

inactive_wheels = [3,5,7]
wheels = [2]

update_velocity( 0 )
	
steering = [4,6]

update_steering( 0 )
	
while (True):

	if ( k.is_pressed('enter') ) :
		#start moving
		update_velocity( 10 )
	elif ( k.is_pressed('s') ) :
		update_velocity( 0 )
	elif ( k.is_pressed('h') ) :
		#turn right
		update_steering( -0.8 )
	elif ( k.is_pressed('l') ) :
		#turn left
		update_steering( 0.8 )
	elif ( k.is_pressed('c') ) :
		#turn forward
		update_steering( 0 )
	elif ( k.is_pressed('1') ) :
		#right - straight - right
		CSC(RIGHT, RIGHT)
	elif ( k.is_pressed('2') ) :
		CSC(LEFT, RIGHT)
	elif ( k.is_pressed('3') ) :
		CSC(RIGHT, LEFT)
	elif ( k.is_pressed('4') ) :
		CSC(LEFT, LEFT)
	elif ( k.is_pressed('5') ) :
		CCC(RIGHT, LEFT, RIGHT)
	elif ( k.is_pressed('6') ) :
		CCC(LEFT, RIGHT, LEFT)

	#steering
	if (useRealTimeSim==0):
		p.stepSimulation()
	time.sleep(0.01)


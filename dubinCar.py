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
VELOCITY = 10
ANGLE = 10.0
TYPE_OF_PATH = ["LSL", "LSR", "RSL", "RSR", "RLR", "LRL" ]

def update_steering ( steeringAngle ) :
	for steer in steering:
		p.setJointMotorControl2(car,steer,p.POSITION_CONTROL,targetPosition=steeringAngle)
	
def update_velocity ( velocity ) :
	for wheel in inactive_wheels:
		p.setJointMotorControl2(car,wheel,p.VELOCITY_CONTROL,targetVelocity=velocity,force=10)

def CSC ( curve1, curve2, time1, time2, time3 ) :
	#start moving
	update_velocity( VELOCITY )
	#turn first curve
	update_steering( curve1 * ANGLE )
	time.sleep( time1 )
	#turn forward
	update_steering( 0 )
	time.sleep( time2 )
	#turn second curve
	update_steering( curve2 * ANGLE )
	time.sleep( time3 )
	#stop moving
	update_velocity( 0 )

def CCC ( curve1, curve2, curve3, time1, time2, time3 ) :
	update_velocity( VELOCITY )
	#turn first curve
	update_steering( curve1 * ANGLE )
	time.sleep( time1 )
	#turn forward
	update_steering( curve2 * ANGLE )
	time.sleep( time2 )
	#turn second curve
	update_steering( curve3 * ANGLE )
	time.sleep( time3 )
	#stop moving
	update_velocity( 0 )

def f(x) :
	return {
		'LSL' : CSC(LEFT, LEFT, time1, time2, time3),
		'LSR' : CSC(LEFT, RIGHT, time1, time2, time3),
		'RSL' : CSC(RIGHT, LEFT, time1, time2, time3),
		'RSR' : CSC(RIGHT, RIGHT, time1, time2, time3),
		'RLR' : CCC(RIGHT, LEFT, RIGHT, time1, time2, time3),
		'LRL' : CCC(LEFT, RIGHT, LEFT, time1, time2, time3),
	} [x]


q0 = (0, 0, 0)
q1 = (40, 0, 3.14)
turning_radius = 1.0
path = dubins.shortest_path(q0, q1, turning_radius)
typeP = TYPE_OF_PATH[path.path_type()]
total_length = path.path_length()
length1 = path.segment_length(0)
length2 = path.segment_length(1)
length3 = path.segment_length(2)

total_time = total_length / VELOCITY
time1 = length1 * total_time / total_length
time2 = length2 * total_time / total_length
time3 = length3 * total_time / total_length

print('')
print( 'shortest path from', q0, ' to ', q1 )
print( 'type of path is: ', TYPE_OF_PATH[path.path_type()] )
print( 'length of shortest path:',  path.path_length() )
print( 'length of first segment: ', path.segment_length(0) )
print( 'length of second segment: ', path.segment_length(1) )
print( 'length of third segment: ', path.segment_length(2) )
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

f(typeP)
	
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

	elif ( k.is_pressed('q') ) :
		p.setJointMotorControl2(car,2,p.POSITION_CONTROL,positionGain=10)

	#steering
	if (useRealTimeSim==0):
		p.stepSimulation()
	time.sleep(0.01)


import pybullet as p 

def printJointInfo(body_id, client_id=0): 
    print(f'Joint info for body id {body_id}\n-----------------------')
    joint_types = ['REVOLUTE', 'PRISMATIC', 'SPHERICAL', 'PLANAR', 'FIXED']
    print('Index Type Name Parent_Link_Name')
    for joint_num in range(p.getNumJoints(body_id, client_id)):
        info = p.getJointInfo(body_id, joint_num, client_id)
        print(f'{joint_num} {joint_types[info[2]]} {info[1]} {info[12]}')

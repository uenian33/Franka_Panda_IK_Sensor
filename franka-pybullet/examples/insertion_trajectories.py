import sys
sys.path.append('../src')

import pybullet as p
import time
from math import sin
import numpy as np
import pickle
from panda import Panda
import transformation

duration = 3000
stepsize = 1e-3

robot = Panda(stepsize)
robot.setControlMode("position")

cat_trajs = np.load('../franka_panda_insertion_logs/experts/expert_cartesian_poses.npy', allow_pickle=True)
j_trajs = np.load('../franka_panda_insertion_logs/experts/expert_joints_poses.npy', allow_pickle=True)

euler_trajs = []
#"""
for tid, jt in enumerate(j_trajs):
    et = []
    for jid, j_pos in enumerate(jt):
        #j_pos = np.append(j_pos, [0,0])
        #print(robot.joints)
        #print(j_pos)
        euler_pose = robot.step(j_pos)
        et.append(euler_pose)

        print('real', robot.getEEStates()[1])
        print('logs', cat_trajs[tid][jid][3:])
    euler_trajs.append(et)

euler_trajs = np.array(euler_trajs)
np.save('../franka_panda_insertion_logs/experts/expert_euler_poses.npy', euler_trajs)

        #euler = transformation.euler_from_quaternion(robot.getEEStates()[1])
        #print(robot.getEEStates_euler())
        #print(transformation.quaternion_from_euler(euler[0], euler[1], euler[2]))
#"""
cat_trajs = np.load('../franka_panda_insertion_logs/experts/expert_euler_poses.npy', allow_pickle=True)


for tid, ct in enumerate(cat_trajs):
    for cid, cat_pos in enumerate(ct):
        target_pos = robot.solveInverseKinematics(cat_pos[:3],transformation.quaternion_from_euler(cat_pos[3], cat_pos[4], cat_pos[5]))#[1,0,0,0])
        robot.step(target_pos)
        
        print('real', robot.getEEStates()[1])
        print('logs', cat_pos[3:])

        time.sleep(0)

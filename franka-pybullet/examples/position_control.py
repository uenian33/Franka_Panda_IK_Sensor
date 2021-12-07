import sys
sys.path.append('../src')

import pybullet as p
import time
from math import sin
import numpy as np
import pickle
from panda_imitation import Panda
import transformation

def test_cartesian_trajs():
    duration = 3000
    stepsize = 1e-3

    robot = Panda(stepsize)
    robot.setLimitations()
    robot.setControlMode("position")

    cat_trajs = np.load('../franka_panda_insertion_logs/experts/expert_euler_poses.npy', allow_pickle=True)

    """
    print(cat_trajs.shape)
    print(np.vstack(cat_trajs).shape)
    print(np.amax(np.vstack(cat_trajs), axis=0))
    print(np.amin(np.vstack(cat_trajs), axis=0))
    c()
    """
    end_poses = []
    for tid, ct in enumerate(cat_trajs):
        ct[-1][3:] = ct[-1][3:] /np.pi*180
        print(ct[-1][:])
        end_poses.append(ct[-1])
    #a()
    #print(end_poses)

    for _ in range(10):
        delt_ang = 3#np.random.uniform(-1, 1) * 20
        delt_x = 0.0#np.random.uniform(-1, 1) * 0.01
        delt_y = 0.0#np.random.uniform(-1, 1) * 0.02
        for tid, ct in enumerate(cat_trajs):
            ct = ct[::2]
            robot.reset()
            print('----------------------------------------------')
            for cid, cat_pos in enumerate(ct):
                robot.step(cat_pos)
                pos = robot.getEEStatesEuler()
                print(pos[0]-cat_pos[0], pos[1]-cat_pos[1], (pos[3]-cat_pos[3])/np.pi*180)
                print(delt_x, delt_y, delt_ang)

            for cid, _ in enumerate(ct):
                cat_pos = ct[-1]
                cat_pos[0] = cat_pos[0] + delt_x
                cat_pos[1] = cat_pos[1] + delt_y
                cat_pos[3] = cat_pos[2] + delt_ang / 180 * np.pi
                robot.step(cat_pos)
                pos = robot.getEEStatesEuler()
                print(pos[0]-cat_pos[0], pos[1]-cat_pos[1], (pos[3]-cat_pos[3])/np.pi*180)
                print(delt_x, delt_y, delt_ang)

    for tid, ct in enumerate(cat_trajs):
        ct = ct[::10]
        robot.reset()
        for cid, cat_pos in enumerate(ct):
            robot.step(cat_pos)
            print(cid)
            time.sleep(1)
    #"""

def load_joint_trajs():
    duration = 3000
    stepsize = 1e-3

    robot = Panda(stepsize)
    robot.setLimitations()
    robot.setControlMode("torque")
    robot.reset()
    robot.setControlMode("torque")

    j_trajs = np.load('../franka_panda_insertion_logs/experts/expert_joints_poses.npy', allow_pickle=True)

    joints = np.vstack(j_trajs)

    jmax = np.amax(joints, axis=0)
    jmin = np.amin(joints, axis=0)
    print(jmax)
    print(jmin)

    jmean = (jmax+jmin)/2.

    for jid, j in enumerate(joints):
        #robot.reset()
        delt = np.random.uniform(-1,1, size=jmean.shape)*(jmax - jmean)
        print(jmax, jmin, jmean, jmax - jmean)
        target_torque =  j #+ delt

        robot.stepTorque(target_torque)
        quat_pose = robot.getEEStatesQuaternion()
        euler_pose = robot.getEEStatesEuler()
        print(quat_pose)
      
        



    
def load_expert_buffer():
    with open('../franka_panda_insertion_logs/expert_replay.pkl', 'rb') as f:
        exp_buffer = pickle.load(f)

#load_joint_trajs()
#a()
test_cartesian_trajs()
a()
load_expert_buffer()  

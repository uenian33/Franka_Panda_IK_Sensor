import sys
sys.path.append('../src')

import time
from math import sin

from panda import Panda
import numpy as np

duration = 30
stepsize = 1e-3

robot = Panda(stepsize)
robot.setControlMode("torque")

j_trajs = np.load('../franka_panda_insertion_logs/experts/expert_joints_poses.npy', allow_pickle=True)
joints = np.vstack(j_trajs)
jmax = np.amax(joints, axis=0)
jmin = np.amin(joints, axis=0)
jmean = (jmax+jmin)/2.

robot.reset()
robot.setControlMode("torque")

for _ in range(10000):
    for j in joints:
        j = j + np.random.uniform(-1,1, size=jmean.shape)*(jmax - jmean)

        target_torque = j

        robot.step(target_torque)


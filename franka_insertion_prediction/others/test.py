#!/usr/bin/env python3


from tensorboardX import SummaryWriter
from itertools import count
import torch
import argparse
import numpy as np
import os
import time
from time import gmtime, strftime

import threading as th
import glob
from datetime import datetime
import glob
import pickle

"""
class PriorInfo(object):
    #docstring for PriorInfo
    def __init__(self, dataset_path="/home/alex/catkin_ws/datasets/datasets_v1"):
        super(PriorInfo, self).__init__()
        self.dataset_path = dataset_path
        self.joint_angles = []
        self.joint_next_angles = []
        self.joint_actions = []


    def load_demonstration(self):
        #dataset_path = "/home/alex/catkin_ws/src/cobot_controllers/datasets/without_extra_grabber/npys/"
        #dataset_path = "/home/alex/catkin_ws/datasets/datasets_v0"
        dataset_path = self.dataset_path

        traj_id = 0
        exp_trajs = []
        traj = None
        sample_ratio = 5

        filename_list = sorted(glob.glob(dataset_path+'/*.npy'))

        for filename in filename_list:
            if filename.endswith(".npy"):# and mode in filename:
                exp_traj = np.load(os.path.join(dataset_path, filename))[:-10][::sample_ratio]    
                exp_trajs.append(exp_traj)

        for exp_t in exp_trajs:
            actions = []
            for tid in range(len(exp_t)-1):
                delta = exp_t[tid+1] - exp_t[tid]
                actions.append(delta)

                self.joint_angles.append(exp_t[tid])
                self.joint_actions.append(delta)
                self.joint_next_angles.append(exp_t[tid+1])

        self.joint_angles = np.array(self.joint_angles)
        self.joint_next_angles = np.array(self.joint_next_angles)
        self.joint_actions = np.array(self.joint_actions)

        return exp_trajs[0:]

    def retrieve_closest_prior(self, joints):
        norms = np.linalg.norm(self.joint_angles - joints, axis=1)
        closest_id = np.argmin(norms)

        return self.joint_angles[closest_id], self.joint_actions[closest_id], self.joint_next_angles[closest_id]
"""
      

class PriorInfo(object):
    """docstring for PriorInfo"""
    def __init__(self, dataset_path="/home/alex/Documents/franka_panda_insertion_logs/experts/"):
        super(PriorInfo, self).__init__()
        self.dataset_path = dataset_path
        self.joint_angles = []
        self.joint_next_angles = []
        self.cartesian_poses = []
        self.cartesian_next_poses = []
        self.joint_actions = []
        self.sample_ratio = 10

    def load_demonstration(self, load_ids=[1]):
        dataset_path = self.dataset_path

        exp_joint_trajs = []
        traj = None
        sample_ratio = self.sample_ratio

        exp_cartesian_trajs = np.load(self.dataset_path+'expert_cartesian_poses.npy', allow_pickle=True)
        exp_joint_trajs = np.load(self.dataset_path+'expert_joints_poses.npy', allow_pickle=True)

        exp_subsample_c_trajs = exp_cartesian_trajs[load_ids].copy()
        exp_subsample_j_trajs = exp_joint_trajs[load_ids].copy()

        for eid, exp_t in enumerate(exp_subsample_j_trajs):
            actions = []
            exp_c_t = exp_cartesian_trajs[eid]

            exp_subsample_j_trajs[eid] = exp_t[::sample_ratio]
            #exp_subsample_c_trajs[eid] = exp_c_t[::sample_ratio]

            for tid in range(len(exp_t)-sample_ratio):
                delta = exp_t[tid+sample_ratio] - exp_t[tid]

                self.joint_angles.append(exp_t[tid])
                self.joint_next_angles.append(exp_t[tid+sample_ratio])
                self.joint_actions.append(delta)
                #self.cartesian_poses.append(exp_c_t[tid])
                #self.cartesian_next_poses.append(exp_c_t[tid+sample_ratio])

        self.joint_angles = np.array(self.joint_angles)
        self.joint_next_angles = np.array(self.joint_next_angles)
        self.joint_actions = np.array(self.joint_actions)
        #self.cartesian_poses = np.array(self.cartesian_poses)
        #self.cartesian_next_poses = np.array(self.cartesian_next_poses)

        return exp_subsample_j_trajs#, exp_subsample_c_trajs

    def retrieve_closest_prior(self, inputs,
                                dis_type='cartesian'):
        if dis_type=='joints':
            norms = np.linalg.norm(self.joint_angles - inputs, axis=1)
            closest_id = np.argmin(norms)
        elif dis_type=='cartesian':
            norms = np.linalg.norm(self.cartesian_poses - inputs, axis=1)
            closest_id = np.argmin(norms)

        return closest_id, self.joint_angles[closest_id], self.joint_actions[closest_id], self.joint_next_angles[closest_id]#, self.cartesian_next_poses[closest_id]   



all_acts = np.load('/home/alex/Documents/franka_panda_insertion_logs/expert_action_under_prior.npy', allow_pickle=True)

# initiate the expert prior class, which contains the expert state-action-next_state in joint angle space
prior = PriorInfo()
# load expert joint trajectory
demos = prior.load_demonstration()

for d in demos:
    print(len(d))


for exp_ep in range(len(demos)):
    demo_traj = demos[exp_ep]
    for ts in range(len(demo_traj)-1):
        curr_joints = demo_traj[ts]
        cid_j, _, _, prior_target_joints = prior.retrieve_closest_prior(curr_joints, dis_type='joints')

        target_joint = demo_traj[ts+1] 
        print(cid_j)
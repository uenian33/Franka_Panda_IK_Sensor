import numpy as np
import torch
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils.models import MixtureDensityNetwork, SimpleNet, DenseNet, SelectionNet
from os.path import dirname, abspath, join

import pinocchio

# pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# differentiable-robot-model
from differentiable_robot_model.robot_model import DifferentiableRobotModel

from data import IKDataset 
from data.data_generator import generate_data
from data.data_config import DataGenConfig

import torch_optimizer as optim

# training hyperparameters
BATCH_SIZE = 2048
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
MOMENTUM = 0.95
LEARNING_RATE_DECAY = 0.97
n_components=30
IN_DIM=7
OUT_DIM=7
WEIGHT_PATH="weights/mdn_ik_weights.pth"
#LEARNING_RATE_DECAY = 3e-6

def dataset_generation():
    # generate dataset
    print("--------------generate Data--------------")
    generate_data()
    print("--------------Loading Data--------------")
    dataset = IKDataset()
    # shuffle=False because each data point is already randomly sampled
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("--------------Data  Loaded--------------\n")
    return train_loader, dataset

def cvae_fk_loss(joint_config: torch.Tensor, true_pose: torch.Tensor, 
                 robot_model: DifferentiableRobotModel) -> torch.Tensor:
    pose = torch.cat(robot_model.compute_forward_kinematics(joint_config, "panda_link7"), axis=1)
    # reconstruction loss in task space
    recon_loss = nn.functional.mse_loss(true_pose*100, pose*100, reduction="sum")
    return recon_loss


def test():
    # device = torch.device("cpu")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = SimpleNet(IN_DIM, OUT_DIM)
    
    #model = MixtureDensityNetwork(IN_DIM, OUT_DIM, n_components=n_components)
    model = SimpleNet(IN_DIM, OUT_DIM)

    model.load_state_dict(torch.load(WEIGHT_PATH))
    model.eval()

    cat_trajs = np.load('data/franka_panda_insertion_logs/experts/expert_cartesian_poses.npy', allow_pickle=True)
    cat_trajs = np.vstack(cat_trajs)

    j_trajs = np.load('data/franka_panda_insertion_logs/experts/expert_joints_poses.npy', allow_pickle=True)
    j_trajs = np.vstack(j_trajs)

    for pid, pos in enumerate(cat_trajs):
        pose = torch.Tensor([pos])
    
        q = model.sample(pose)[0].detach().numpy()

        print("Generated q: ", q)
        print("desiserd  q:", j_trajs[pid])


        robot_path = "resources/" + DataGenConfig.ROBOT
        urdf_path = robot_path + "/urdf/"+DataGenConfig.ROBOT_URDF
        # setup robot model and data
        robot = pinocchio.buildModelFromUrdf(urdf_path)
        data = robot.createData()
        # setup end effector
        ee_name = DataGenConfig.EE_NAME
        ee_link_id = robot.getFrameId(ee_name)
        # joint limits (from urdf)
        lower_limit = np.array(robot.lowerPositionLimit)
        upper_limit = np.array(robot.upperPositionLimit)
        pinocchio.framesForwardKinematics(robot, data, q)
        desired_pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])

        print("Desired Pose", pose[:].cpu().numpy())
        print("Generated Pose: ", desired_pose[:])
        print("Error: ", np.linalg.norm(pos[:3] - desired_pose[:3]))


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    robot_path = "resources/" + DataGenConfig.ROBOT
    urdf_path = robot_path + "/urdf/"+DataGenConfig.ROBOT_URDF
    #urdf_path = "resources/franka/urdf/panda_arm.urdf"
    robot = DifferentiableRobotModel(
        urdf_path, name="franka_panda", device=str(device)
    )

    # device = torch.device("cpu")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = SimpleNet(IN_DIM, OUT_DIM)
    #model = MixtureDensityNetwork(IN_DIM, OUT_DIM, n_components=n_components)
    model = SimpleNet(IN_DIM, OUT_DIM)

    # optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LEARNING_RATE_DECAY)
    beta = 0.02
    #"""
    optimizer = optim.Yogi(
        model.parameters(),
        lr= 1e-2,
        betas=(0.9, 0.999),
        eps=1e-3,
        initial_accumulator=1e-6,
        weight_decay=0,
    )
    optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LEARNING_RATE_DECAY)
    #"""

    # setup differentiable robot model stuff
    iter_counts=0

    # training loop
    print("----------------Training----------------")

    iter_counts = 0
    for epoch in range(NUM_EPOCHS):
        train_loader, dataset = dataset_generation()
        epoch_error = 0
        for pose, joint_config in train_loader:
            mseloss = torch.nn.MSELoss()
            preds = model(pose)
            #loss = model.loss(pose, joint_config).mean()
            loss = 0
          
            #weighted_preds = model.weighted_forward(pose)
            loss += cvae_fk_loss(preds, pose, robot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_error += loss.item()
            
            iter_counts+=1
            if iter_counts%100==0:
                preds = model.sample(pose)
                #test_err = np.square(np.subtract(preds.numpy(), joint_config.numpy())).mean()

                print("iter Number: {} || Average Error: {}".format(iter_counts, loss.item()))
                #logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}")

        #if epoch > 25:
        #    optimizer_scheduler.step()
        print("Epoch Number: {} || Average Error: {}".format(epoch, epoch_error/dataset.n_samples))
        torch.save(model.state_dict(), WEIGHT_PATH)

if __name__ == "__main__":
    #test()
    train()



# pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# differentiable-robot-model
from differentiable_robot_model.robot_model import DifferentiableRobotModel

from model import ConstrainedCVAE
from data import IKDataset 
from data.data_generator import generate_data

import torch_optimizer as optim

# training hyperparameters
BATCH_SIZE = 512
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
MOMENTUM = 0.95
LEARNING_RATE_DECAY = 0.97
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
                 mean: torch.Tensor, log_variance: torch.Tensor, 
                 robot_model: DifferentiableRobotModel, beta: float = 0.02) -> torch.Tensor:
    pose = torch.cat(robot_model.compute_forward_kinematics(joint_config, "panda_link7"), axis=1)
    # reconstruction loss in task space
    recon_loss = nn.functional.mse_loss(true_pose, pose, reduction="sum")
    kl_loss = 0.5 * torch.sum(log_variance.exp() + mean.pow(2) - 1. - log_variance)
    return recon_loss + beta*kl_loss


def cvae_loss(joint_config: torch.Tensor, true_joint_config: torch.Tensor, 
              mean: torch.Tensor, log_variance: torch.Tensor, 
              beta: float = 0.01) -> torch.Tensor:
    # reconstruction loss in configuration space
    recon_loss = nn.functional.mse_loss(joint_config, true_joint_config, reduction="sum")
    kl_loss = 0.5 * torch.sum(log_variance.exp() + mean.pow(2) - 1. - log_variance)
    return recon_loss + beta*kl_loss

def train():
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvae = ConstrainedCVAE().to(device)
    cvae.train()

    

    # optimizer
    optimizer = torch.optim.Adam(cvae.parameters(), lr=LEARNING_RATE)
    optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LEARNING_RATE_DECAY)
    beta = 0.02
    #"""
    optimizer = optim.Yogi(
        cvae.parameters(),
        lr= 1e-2,
        betas=(0.9, 0.999),
        eps=1e-3,
        initial_accumulator=1e-6,
        weight_decay=0,
    )
    optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LEARNING_RATE_DECAY)
    #"""

    # setup differentiable robot model stuff
    urdf_path = "resources/franka/urdf/panda_arm.urdf"
    robot_model = DifferentiableRobotModel(
        urdf_path, name="franka_panda", device=str(device)
    )
    iter_counts=0


    # training loop
    print("----------------Training----------------")
    for epoch in range(NUM_EPOCHS):
        train_loader, dataset = dataset_generation()
        epoch_error = 0
        for pose, joint_config in train_loader:
            pose = pose.to(device)
            joint_config = joint_config.to(device)
            joint_config_pred, mean, log_variance = cvae(pose, joint_config)
            loss = cvae_loss(joint_config_pred, joint_config,
                                mean, log_variance, beta)
            #loss = cvae_fk_loss(joint_config_pred, pose, 
            #                    mean, log_variance, robot_model, beta)
            epoch_error += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(loss.item())
            iter_counts+=1
            if iter_counts%100==0:
                print("iter Number: {} || Average Error: {}".format(iter_counts, loss.item()))

        #if epoch > 25:
        #    optimizer_scheduler.step()
        print("Epoch Number: {} || Average Error: {}".format(epoch, epoch_error/dataset.n_samples))
        torch.save(cvae.state_dict(), "model/weights/constrained_cvae_weights.pth")

    print("-----------Training  Completed-----------")

    # save the weights
    torch.save(cvae.state_dict(), "model/weights/constrained_cvae_weights.pth")
    

if __name__ == "__main__":
    train()

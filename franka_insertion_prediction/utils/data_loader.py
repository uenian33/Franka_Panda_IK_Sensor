import torch
from torchvision import datasets
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import utils.transformation as transformation

class TargetPoseSequenceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(path, 'rb') as f:
            self.expert_data = pickle.load(f)

        
        self.ys = []
        self.xs = []
        self.force = []

        self.load_data()

    def load_data(self):
        expert_infos = self.expert_data[4]

        for exp_info_traj in expert_infos:
            for info in exp_info_traj[:]: 
                #print(info)
                #a()
                ext_f = np.array(info[-2])
                s = np.array(info[0])
                a = np.array(info[1])
                x = np.stack([s,a])#.flatten()

                position = exp_info_traj[-1][2]
                #print(position)
                pose = exp_info_traj[-1][3]
                euler = transformation.euler_from_quaternion(pose)
                euler = np.array(euler) / np.pi * 1800
                y = np.array([position[0]*1000, position[1]*1000, euler[0]]).astype(np.float64)

                #print(pose)

                self.ys.append(y)
                self.xs.append(x)
                self.force.append(ext_f)
        print(self.ys)

        self.xs = torch.tensor(np.array(self.xs))
        self.ys = torch.tensor(np.array(self.ys))
        self.force = torch.tensor(np.array(self.force))

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.xs[idx].float(), self.force[idx].float(), self.ys[idx].float()


class TargetPoseDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path, configs):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(path, 'rb') as f:
            self.expert_data = pickle.load(f)

        self.configs = configs
        self.ys = []
        self.xs = []

        self.raw_ys = []
        self.raw_xs = []

        self.load_data()

    def load_data(self):
        if not self.configs['resume']:
            self.raw_xs = []
            self.raw_ys = []
            expert_infos = self.expert_data[4]

            for exp_info_traj in expert_infos:
                for info in exp_info_traj[:]: 
                    # use previous ee pose, ee velocities, ee wrench for task laerning
                    previous_ee_pose_quat = info['previous_ee_pose_quat']
                    previous_ee_pose_euler = transformation.euler_from_quaternion(previous_ee_pose_quat[3:])
                    previous_ee_pose = np.append(previous_ee_pose_quat[:3], previous_ee_pose_euler)
                    
                    ee_cartesian_velocities = np.mean(np.array(info['ee_cartesian_velocities']), axis=0)
                    ee_wrench = np.mean(np.array(info['ee_cartesian_wrench_sequence']), axis=0)

                    x = np.stack([previous_ee_pose,ee_cartesian_velocities,ee_wrench])#.flatten()

                    # use x, y, skew as the prediction label
                    position = exp_info_traj[-1]['ee_pose_quat'][:3]
                    pose = exp_info_traj[-1]['ee_pose_quat'][3:]
                    euler = transformation.euler_from_quaternion(pose)
                    euler = np.array(euler) / np.pi * 180
                    y = np.array([position[0]*1000, position[1]*1000, euler[0]]).astype(np.float64)

                    self.raw_ys.append(y)
                    self.raw_xs.append(x)
        else:
            print('load previous generated online data...')

            path = self.configs['online_data_save_path']
            with open(path, 'rb') as data_file:
                online_data = pickle.load(data_file)
            self.raw_xs, self.raw_ys = online_data[0], online_data[1]

        
        self.convert_data()

        return

    def convert_data(self):
        self.ys = self.raw_ys.copy()
        self.xs = self.raw_xs.copy()

        self.xs = torch.tensor(np.array(self.xs))
        self.ys = torch.tensor(np.array(self.ys))
        return 

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.xs[idx].float(), self.ys[idx].float()



def get_loader(args):
    #dataset = TargetPoseDataset(path=args.data_path)
    #test = TargetPoseDataset(path=args.data_path)

    train = TargetPoseDataset(path=args.data_path)
    train.xs = train.xs[:-10]
    train.ys = train.ys[:-10]

    test = TargetPoseDataset(path=args.data_path)
    test.xs = test.xs[-10:]
    test.ys = test.ys[-10:]


    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                 batch_size=1024,
                                                 shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=1024 * 2,
                                                shuffle=False,
                                                drop_last=False)

    return train_loader, test_loader


def get_sequence_loader(args):
    #dataset = TargetPoseDataset(path=args.data_path)
    #test = TargetPoseDataset(path=args.data_path)

    train = TargetPoseSequenceDataset(path=args.data_path)
    train.xs = train.xs[:-10]
    train.ys = train.ys[:-10]
    train.force = train.force[:-10]

    test = TargetPoseSequenceDataset(path=args.data_path)
    test.xs = test.xs[-10:]
    test.ys = test.ys[-10:]
    test.force = test.force[-10:]


    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                 batch_size=1024,
                                                 shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=1024 * 2,
                                                shuffle=False,
                                                drop_last=False)

    return train_loader, test_loader
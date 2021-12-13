import torch
from torchvision import datasets
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import utils.transformation as transformation

def parse_xy(info, pose_quat=None):
    previous_ee_pose_quat = info['previous_ee_pose_quat']
    previous_ee_pose_euler = transformation.euler_from_quaternion(previous_ee_pose_quat[3:])
    previous_ee_pose = np.append(previous_ee_pose_quat[:3], previous_ee_pose_euler)


    ee_pose_quat = info['ee_pose_quat']
    ee_pose_euler = transformation.euler_from_quaternion(ee_pose_quat[3:])
    ee_pose = np.append(ee_pose_quat[:3], ee_pose_euler)

    delta_ee_pose = ee_pose - previous_ee_pose

    
    ee_cartesian_velocities = np.mean(np.array(info['ee_cartesian_velocities']), axis=0)
    ee_wrench = np.mean(np.array(info['ee_cartesian_wrench_sequence']), axis=0)

    x = np.stack([previous_ee_pose,delta_ee_pose,ee_wrench])#.flatten()

    if pose_quat is not None:
        # use x, y, skew as the prediction label
        position = pose_quat[:3]
        pose = pose_quat[3:]
        euler = transformation.euler_from_quaternion(pose)
        euler = np.array(euler) / np.pi * 180
        y = np.array([position[0]*1000, position[1]*1000, euler[0]]).astype(np.float64)

        return x,y
    else:
        return x

class TargetPoseSequenceDataset(Dataset):
    """sequence dataset."""
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

        self.sequence_size = self.configs['sequence_size']

        self.load_data()

    def load_data(self):
        self.raw_xs = []
        self.raw_ys = []

        if not self.configs['resume']:
            expert_infos = self.expert_data[4]

            for exp_info_traj in expert_infos:
                for iid, info in enumerate(exp_info_traj[:-self.sequence_size]): 
                    #print(info)
                    x_ = []
                    y_ = []
                    for e in exp_info_traj[iid:iid+self.sequence_size]:
                        x_meta, y_meta = parse_xy(e, exp_info_traj[-1]['ee_pose_quat'])
                        #print(x_meta)
                        x_.append(x_meta)
                        y_.append(y_meta)

                    x = np.array(x_)
                    y_ = np.array(y_)
                    y = np.array(y_meta)

                    print(y_[0])

                    if (y_[0]==y_[1]).all() and (y_[0]==y_[2]).all():
                        self.raw_ys.append(y)
                        self.raw_xs.append(x)

        else:
            print('load previous generated online data...')
            path = self.configs['online_data_save_path']
            with open(path, 'rb') as data_file:
                online_data = pickle.load(data_file)

            self.raw_xs, self.raw_ys = online_data[0], online_data[1]

            """
            This is the version that load sequence data from single stored dataset
            for iid in range(len(online_data[0])-self.sequence_size):
                x_ = []
                y_ = []
                for eid, x_meta in enumerate(online_data[0][iid:iid+self.sequence_size]):
                    #print(x_meta)
                    x_.append(x_meta)
                    y_.append(online_data[1][iid+eid])

                x = np.array(x_)
                y = np.array(y_[0])
                #print(x[0])

                #print('........')
                #print(y_[0])
                if (y_[0]==y_[1]).all() and (y_[0]==y_[2]).all():
                    self.raw_ys.append(y)
                    self.raw_xs.append(x)
                else:
                    print('not the same trajectory', y)
            """
        
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
                    x, y = parse_xy(info, exp_info_traj[-1]['ee_pose_quat'])

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
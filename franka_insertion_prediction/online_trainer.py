import sys
sys.path.append('/home/alex/catkin_ws/src/panda-insert/src/cobot_controllers/franka_insertion_prediction/')
import argparse
import datetime
import os
import pickle
import torch
import yaml
import numpy as np
from utils.solver import Solver, SequenceSolver
from utils.transformation import quaternion_from_euler, euler_from_quaternion
from utils.transformer import TModel, SimpleNet, TSequenceModel
from utils.data_loader import get_loader, get_sequence_loader, TargetPoseDataset, TargetPoseSequenceDataset
from utils.human_priors import HumanPrior

class OnlineLearner(object):
    """docstring for OnlineLearner"""
    def __init__(self, 
                patch_dim=6, 
                num_patches=3,
                args_path='/home/alex/catkin_ws/src/panda-insert/src/cobot_controllers/franka_insertion_prediction/config/online_config.yaml'):
        super(OnlineLearner, self).__init__()

        f = open(args_path)
        args = yaml.load(f)

        self.args = args
        #os.makedirs(args['model_path'], exist_ok=True)

        if self.args['model_type']=='single':
            self.model = TModel(patch_dim=patch_dim, num_patches=num_patches, out_dim=3, 
                        dim=64, depth=5, heads=10, mlp_dim=64, 
                        pool = 'cls', dim_head = 64, dropout = 0.01, emb_dropout = 0.)
        elif self.args['model_type']=='sequence':
            self.model = TSequenceModel(patch_dim=patch_dim, num_patches=num_patches, out_dim=3, 
                        dim=64, depth=5, heads=10, mlp_dim=64, 
                        pool = 'cls', dim_head = 64, dropout = 0.01, emb_dropout = 0.)
        else:
            print('model type must be either \'single\' or \'sequence\'')

        self.human_priors = HumanPrior(self.args)

        self.train_set = None
        self.test_set = None
        self.train_loader = None
        self.test_loader = None

        self.load_dataset()
        self.solver = Solver(args, self.model, self.train_loader, self.test_loader)

    def load_dataset(self):
        if self.args['model_type']=='single':
            self.train_set = TargetPoseDataset(path=self.args['data_path'], configs=self.args)
            self.test_set = TargetPoseDataset(path=self.args['data_path'], configs=self.args)
          
        elif self.args['model_type']=='sequence':
            self.train_set = TargetPoseSequenceDataset(path=self.args['data_path'], configs=self.args)
            self.test_set = TargetPoseSequenceDataset(path=self.args['data_path'], configs=self.args)

        self.test_set.xs = self.test_set.xs[-10:]
        self.test_set.ys = self.test_set.ys[-10:]


        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set,
                                                     batch_size=1024,
                                                     shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_set,
                                                    batch_size=1024 * 2,
                                                    shuffle=False,
                                                    drop_last=False)  


    def load_model(self):
        self.solver.load_model()
        return

    def online_learn(self, meta_epoch=50, stop_critersia=100):
        training = True
        while training:
            self.solver.train_loader = self.train_loader
            self.solver.test_loader = self.test_loader
            self.solver.train(epochs=meta_epoch, log_epoch=meta_epoch)
            train_acc, test_acc = self.solver.test()

            if train_acc < stop_critersia:
                training = False



    def learn(self, epochs=None):
        self.solver.train_loader = self.train_loader
        self.solver.test_loader = self.test_loader
        self.solver.train(epochs=epochs)

    def choose_action(self, x, real_joints, prior_type='action', nsteps=5):
        """
            prior_type: 'action', 'seg_traj', 'whole_traj'
        """
        target_pos = self.target_prediction(x)

        if prior_type=='action':
            traj = self.human_priors.retrieve_prior_action(target_pos, real_joints)
        if prior_type=='seg_traj':
            traj = self.human_priors.retrieve_prior_segment_trajectory(target_pos, real_joints, segment_size=nsteps)
        if prior_type=='whole_traj':
            traj = self.human_priors.retrieve_prior_whole_trajectory(target_pos)
            
        return traj

    def target_prediction(self, x):

        x = torch.tensor([x]).float()
        target_pos = self.solver.model(x).detach().numpy()[0]

        return target_pos

    def add_training_data(self, infos, cartesian_traj):
        print(len(self.train_set.raw_xs))
        for iid, info in enumerate(infos):
            # generate inputs
            x = self.process_data_to_inputs(info)
            self.train_set.raw_xs.append(x)

            # use x, y, skew as the prediction label
            position = infos[-1]['ee_pose_quat'][:3]
            pose = infos[-1]['ee_pose_quat'][3:]
            euler = euler_from_quaternion(pose)
            euler = np.array(euler) / np.pi * 180
            y = np.array([position[0]*1000, position[1]*1000, euler[0]]).astype(np.float64)

            self.train_set.raw_ys.append(y)

        #print(len(self.train_set.raw_xs), len(cartesian_traj))

        self.train_set.convert_data()

    def process_data_to_inputs(self, info):
        # use previous ee pose, ee velocities, ee wrench for task laerning
        previous_ee_pose_quat = info['previous_ee_pose_quat']
        previous_ee_pose_euler = euler_from_quaternion(previous_ee_pose_quat[3:])
        previous_ee_pose = np.append(previous_ee_pose_quat[:3], previous_ee_pose_euler)
        
        ee_cartesian_velocities = np.mean(np.array(info['ee_cartesian_velocities']), axis=0)
        ee_wrench = np.mean(np.array(info['ee_cartesian_wrench_sequence']), axis=0)

        
        x = np.stack([previous_ee_pose,ee_cartesian_velocities,ee_wrench])#.flatten()
        return x

    def add_human_prior(self, traj, targets):
        self.human_priors.add_prior(traj, targets)
        return

    def save_data(self):
        path = self.args['online_data_save_path']
        with open(path, 'wb') as data_file:
            pickle.dump([self.train_set.raw_xs, self.train_set.raw_ys], data_file)
        return

    def save_model(self):
        self.solver.save_model(self.solver.finalmodel_save_path)

    def save_human_prior(self):
        self.human_priors.save_prior()

    def euler2quat(self, x):
        return quaternion_from_euler(x)

    def quat2euler(self, x):
        return euler_from_quaternion(x)

    
        
        
def test():

    #args['model_path = os.path.join(args['model_path)
    learner = OnlineLearner(args_path='config/config.yaml')
    learner.learn()
    #learner.load_model()
    tmp = 'data/test.npy'
    #learner.save_data(tmp)
    #learner.save_human_prior()

   

def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


if __name__ == '__main__':
    test()

    
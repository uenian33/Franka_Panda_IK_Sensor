import sys
sys.path.append('/home/alex/catkin_ws/src/panda-insert/src/cobot_controllers/franka_insertion_prediction/')
import argparse
import datetime
import os
import pickle
import torch
import yaml
from utils.solver import Solver, SequenceSolver
from utils.transformation import quaternion_from_euler, euler_from_quaternion
from utils.transformer import TModel, SimpleNet, TSequenceModel
from utils.data_loader import get_loader, get_sequence_loader, TargetPoseDataset
from utils.human_priors import HumanPrior

class OnlineLearner(object):
    """docstring for OnlineLearner"""
    def __init__(self, args_path):
        super(OnlineLearner, self).__init__()

        f = open(args_path)
        args = yaml.load(f)

        self.args = args
        os.makedirs(args['model_path'], exist_ok=True)

        self.model = TModel(patch_dim=7, num_patches=3, out_dim=3, 
                    dim=64, depth=5, heads=10, mlp_dim=64, 
                    pool = 'cls', dim_head = 64, dropout = 0.01, emb_dropout = 0.)

        self.human_priors = HumanPrior(args['prior_traj_path'], args['prior_cartesian_path'])

        self.train_set = None
        self.test_set = None
        self.train_loader = None
        self.test_loader = None

        self.load_dataset()
        self.solver = Solver(args, self.model, self.train_loader, self.test_loader)

    def load_dataset(self):
        self.train_set = TargetPoseDataset(path=self.args['data_path'])
        self.train_set.xs = self.train_set.xs[:-10]
        self.train_set.ys = self.train_set.ys[:-10]

        self.test_set = TargetPoseDataset(path=self.args['data_path'])
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


    def learn(self):
        self.solver = Solver(self.args, self.model, self.train_loader, self.test_loader)
        self.solver.train()

    def choose_action(self, x, real_joints, prior_type='action', nsteps=5):
        """
            prior_type: 'action', 'seg_traj', 'whole_traj'
        """
        x = torch.tensor([x])
        target_pos = solver.model(x)

        if prior_type=='action':
            traj = self.human_priors.retrieve_prior_action(target_pos, real_joints)
        if prior_type=='seg_traj':
            traj = self.human_priors.retrieve_prior_segment_trajectory(target_pos, real_joints, segment_size=nsteps)
        if prior_type=='whole_traj':
            traj = self.human_priors.retrieve_prior_whole_trajectory(target_pos)
            
        return traj


    def add_data(self, x, y):
        self.train_set.raw_xs.append(x)
        self.train_set.raw_ys.append(x)
        self.train_set.conver_data()

    def add_human_prior(self, traj, targets):
        self.human_priors.add_prior(traj, targets)
        return

    def save_data(self, path):
        with open(path, 'wb') as data_file:
            pickle.dump([self.train_set.raw_xs, self.train_set.raw_ys], data_file)
        return

    def save_model(self, path):
        self.solver.save_model(path)

    def save_human_prior(self):
        self.human_priors.save_prior()

    def euler2quat(self, x):
        return quaternion_from_euler(x)

    def quat2euler(self, x):
        return euler_from_quaternion(x)

    
        
        
def test():

    #args['model_path = os.path.join(args['model_path)
    learner = OnlineLearner('config/config.yaml')
    #learner.learn()
    learner.load_model()
    tmp = 'data/test.npy'
    learner.save_data(tmp)
    learner.save_human_prior()

   

def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


#if __name__ == '__main__':
#    test()

    
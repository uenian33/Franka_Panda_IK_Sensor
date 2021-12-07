import numpy as np
import pickle
import pybullet as p

class HumanPrior(object):
	"""docstring for HumanPrior"""
	def __init__(self, joint_traj_path='../franka_panda_insertion_logs/experts/expert_joints_poses.npy',
					   cartesian_traj_path='../franka_panda_insertion_logs/experts/expert_joints_poses.npy'):
		self.joint_traj_path = joint_traj_path
		self.cartesian_traj_path = cartesian_traj_path
		self.joint_trajs = []
		self.cartesian_trajs = []
		self.targets = []

	def load(self):
		self.joint_trajs = np.load(self.joint_traj_path, allow_pickle=True).tolist()
		self.cartesian_trajs = np.load(self.cartesian_traj_path, allow_pickle=True).tolist()

		# code_priors:
		for jid, jtraj in enumerate(self.joint_trajs):
			self.targets = cartesian_trajs[jid][-1]

	def add_prior(self, traj, tar):
		self.targets.append(tar)
		self.joint_trajs.append(traj)

	def save_prior(self):
		np.save(self.joint_traj_path.split('.')[0]+'_new', np.array(self.joint_trajs))
		np.save(self.cartesian_traj_path.split('.')[0]+'_new', np.array(self.cartesian_trajs))


	def retrieve_prior_action(self, preds, real_joints):
		targets = np.array(targets)
		closest_id = np.amin(np.linalg.norm(targets-preds))

		likely_traj = self.joint_trajs[closest_id]

		real_joints = np.array(real_joints)
		likely_traj = np.array(likely_traj)
		a_id = np.amin(np.linalg.norm(likely_traj-real_joints))

		# return the next action of the most similar action from the most likely trajectory
		try: 
			action = likely_traj[a_id+1]
		except:
			action = likely_traj[a_id]

		return action

	def retrieve_prior_segment_trajectory(self, preds, real_joints, segment_size=5):
		targets = np.array(targets)
		closest_id = np.amin(np.linalg.norm(targets-preds))

		likely_traj = self.joint_trajs[closest_id]

		real_joints = np.array(real_joints)
		likely_traj = np.array(likely_traj)
		a_id = np.amin(np.linalg.norm(likely_traj-real_joints))

		# return the next action of the most similar action from the most likely trajectory
		actions = likely_traj[a_id : a_id+segment_size]

		return actions


	def retrieve_prior_whole_trajectory(self, preds):
		targets = np.array(targets)
		closest_id = np.amin(np.linalg.norm(targets-preds))

		return self.joint_trajs[closest_id]



		
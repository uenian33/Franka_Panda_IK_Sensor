import numpy as np
import pickle
import pybullet as p
import sys
sys.path.append('/home/alex/catkin_ws/src/panda-insert/src/cobot_controllers/franka_insertion_prediction/')
from utils.transformation import quaternion_from_euler, euler_from_quaternion

class HumanPrior(object):
	"""docstring for HumanPrior"""
	def __init__(self, 
			   configs):

		self.configs=configs
		self.joint_trajs = []
		self.cartesian_trajs = []
		self.targets = []
		self.load()

	def convert_pose_to_target(self, pose):
		x = pose[0]
		y = pose[1]
		quat = pose[3:] 
		euler = euler_from_quaternion(quat)
		#print([x,y,euler[0] / np.pi * 180])
		tar = np.array([x,y,euler[0]/np.pi * 180])
		return tar

	def load(self):
		if not self.configs['resume']:
			joint_trajs = np.load(self.configs['prior_traj_path'], allow_pickle=True).tolist()
			self.cartesian_trajs = np.load(self.configs['prior_cartesian_path'], allow_pickle=True).tolist()

			# code_priors:
			for jid, jtraj in enumerate(joint_trajs):
				self.joint_trajs.append(jtraj[:])
				cartesian_pose = self.cartesian_trajs[jid][-1]
				tar = self.convert_pose_to_target(cartesian_pose)
				self.targets.append(tar)
				#a()
		else:
			joint_trajs = np.load(self.configs['prior_traj_path'], allow_pickle=True).tolist()
			self.targets = np.load(self.configs['prior_targets_path'], allow_pickle=True).tolist()
			self.cartesian_trajs = np.load(self.configs['prior_cartesian_path'], allow_pickle=True).tolist()

			# code_priors:
			for jid, jtraj in enumerate(joint_trajs):
				self.joint_trajs.append(jtraj[:])


		print(self.targets)

	def add_prior(self, traj, cartesian):
		pose = cartesian[-1]
		tar = self.convert_pose_to_target(pose)
		self.targets.append(tar)
		self.joint_trajs.append(traj)
		self.cartesian_trajs.append(cartesian)

	def save_prior(self):
		if self.configs['resume']:
			np.save(self.configs['prior_traj_path'], self.joint_trajs)
			np.save(self.configs['prior_cartesian_path'], self.cartesian_trajs)
			np.save(self.configs['prior_targets_path'], self.targets)
		else:
			j_path = self.configs['prior_traj_path'].split('.')[0]+'_new'
			j_path = j_path.replace('trajectories', 'online')
			c_path = self.configs['prior_cartesian_path'].split('.')[0]+'_new'
			c_path = c_path.replace('trajectories', 'online')
			np.save(j_path, self.joint_trajs)
			np.save(c_path, self.cartesian_trajs)
			np.save(self.configs['prior_targets_path'], self.targets)


	def retrieve_prior_action(self, preds, real_joints):
		targets = np.array(self.targets)
		print(targets, preds)
		closest_id = np.argmin(np.linalg.norm(targets-preds))

		likely_traj = self.joint_trajs[closest_id]
		#print(likely_traj)
		#a()

		real_joints = np.array(real_joints)
		likely_traj = np.array(likely_traj)
		a_id = np.argmin(np.linalg.norm(likely_traj-real_joints))
		print(a_id, likely_traj.shape)

		# return the next action of the most similar action from the most likely trajectory
		try: 
			action = likely_traj[a_id+1]
		except:
			action = likely_traj[a_id]

		return action

	def retrieve_prior_segment_trajectory(self, preds, real_joints, sample_ratio=5, segment_size=5):
		targets = np.array(self.targets)
		closest_id = np.argmin(np.linalg.norm(targets-preds))

		likely_traj = self.joint_trajs[closest_id][::sample_ratio]

		real_joints = np.array(real_joints)
		likely_traj = np.array(likely_traj)
		a_id = np.argmin(np.linalg.norm(likely_traj-real_joints))

		# return the next action of the most similar action from the most likely trajectory
		actions = likely_traj[a_id : a_id+segment_size]

		return actions


	def retrieve_prior_whole_trajectory(self, preds, sample_ratio=5):
		targets = np.array(self.targets)
		closest_id = np.argmin(np.linalg.norm(targets-preds))
		print(preds, targets)

		return self.joint_trajs[closest_id][::sample_ratio]



		
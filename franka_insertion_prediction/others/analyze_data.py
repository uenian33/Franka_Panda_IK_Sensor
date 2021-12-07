import pickle
import numpy as np 
import matplotlib.pyplot as plt
import transformation

def visualize_ext_f():
	with open('expert_data.pkl', 'rb') as f:
		expert_data = pickle.load(f)

	#print(expert_data[0]) # action
	#print(expert_data[1]) # state
	#print(expert_data[2][0]) # next state
	#print(expert_data[3][0]) # done
	#print(expert_data[4][0]) # info

	for exp_data in expert_data[4]:
		ext_f = []
		for i in exp_data: 
			#ext_f.append(np.mean(np.array(i[-1]), axis=0))
			ext_f.append(np.array(i[-1]))

		ext_f = np.vstack (ext_f)
		plt.plot(ext_f[:,6])
		plt.show()

def supervised_data():
	with open('expert_data.pkl', 'rb') as f:
		expert_data = pickle.load(f)

	for d in expert_data[4][0][-1]:
		print(d)
	euler = transformation.euler_from_quaternion(expert_data[4][0][-1][3])
	#print(np.array(euler) / np.pi * 180)
	euler = transformation.euler_from_quaternion(expert_data[4][1][-1][3])
	#print(np.array(euler) / np.pi * 180)
	euler = transformation.euler_from_quaternion(expert_data[4][2][-1][3])
	#print(np.array(euler) / np.pi * 180)
	euler = transformation.euler_from_quaternion(expert_data[4][3][-1][3])
	#print(np.array(euler) / np.pi * 180)
	euler = transformation.euler_from_quaternion(expert_data[4][4][-1][3])
	#print(np.array(euler) / np.pi * 180)
	#a()

	#print(expert_data[4][4][-1][3])
	ys = []
	xs = []

	expert_infos = expert_data[4]

	for exp_info_traj in expert_infos:
		for info in exp_info_traj[:]: 
			#print(info)
			#a()
			ext_f = np.mean(np.array(info[-1]), axis=0)
			s = np.array(info[0])
			a = np.array(info[1])
			y = np.stack([s,a,ext_f])#.flatten()

			position = exp_info_traj[-1][2]
			#print(position)
			pose = exp_info_traj[-1][3]
			euler = transformation.euler_from_quaternion(pose)
			euler = np.array(euler) / np.pi * 180
			x = np.array([position[0], position[1], euler[0]])

			#print(pose)
			print(y.shape)			

			ys.append(y)
			xs.append(y)

	return xs, ys

supervised_data()	
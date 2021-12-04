import pickle
import numpy as np 
import matplotlib.pyplot as plt

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

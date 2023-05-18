import numpy as np
import os
resolution = np.array([60]) 

for i1 in resolution:
	IFN = [f for f in os.listdir(r'./') if (f.startswith(f'activate-mrg{i1}'))&(f.endswith('.npy'))]	

	OP_Dictionary = dict()
	for input_filename in IFN:
		output_dict = np.load(input_filename,allow_pickle='TRUE')
		for key in output_dict.item():
			value = np.squeeze(output_dict.item().get(key)).reshape(-1,1)   
			if key in OP_Dictionary:
				OP_Dictionary[key] = np.vstack((OP_Dictionary[key], value))
			else:
				OP_Dictionary[key] = value
			
	np.save(f'ACTIVATE_DataRetrievals_{i1}.npy', OP_Dictionary) 
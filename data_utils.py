import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from mat73 import loadmat
import pandas as pd
import copy
import mri_utils
from fft_utils import numpy_2_complex
import numpy as np
import os

#Returns zero-paded array (2D)
def pad_array_center(array, target_shape):
	pad_shape = [(int((target_i - array_i)/2), target_i - array_i - int((target_i - array_i)/2)) 
				for array_i, target_i in zip(array.shape, target_shape)]
	
	padded_array = np.pad(array, pad_shape, mode='constant', constant_values=0)
	return padded_array

# Single Channel Cine
class SCCineDataset(Dataset):
	""" MRI knee dataset with k-space raw data, coil sensitivities and sampling mask
	Adapted from Hammernik et al """
	def __init__(self, **kwargs):
		"""
		Parameters:
		root_dir: str
			root directory of data
		dataset_name: list of str
			list of directory to load data from
		transform: 
		"""
		options = {}

		for key in kwargs.keys():
			options[key] = kwargs[key]

		self.options = options
		self.root_dir = Path(self.options['root_dir'])
		xlsx_file = self.options['xlsx_file']
		# Processing directory
		original_df = pd.read_excel(xlsx_file)

		# Load raw data and coil sensitivities name
		if options['mode'] == 'train':
			datatype_key  = 'TrainingSet'
			patient_dataframe = original_df.loc[(original_df['trainVal'] == datatype_key) & (original_df['AF'] != 'FullSample')]
		elif options['mode'] == 'eval':
			datatype_key = 'ValidationSet'
			patient_dataframe = original_df.loc[original_df['trainVal'] == datatype_key]

		self.patient_dataframe = patient_dataframe

	def __len__(self):
		return len(self.patient_dataframe)



	def __getitem__(self,idx):
		# pytorch recommended special case handling
		if torch.is_tensor(idx):
			idx = idx.tolist()

		#Params and path setup
		options = self.options
		root_dir = self.root_dir
		df = self.patient_dataframe
		kspace_dir = 	os.path.join(root_dir, df.iloc[idx]['kspacePath'])
		mask_dir = 		os.path.join(root_dir, df.iloc[idx]['maskPath'])
		slice = 		df.iloc[idx]['slice'] - 1 #Zero based
		timeFrame = 	df.iloc[idx]['timeFrame'] - 1

		if options['mode'] == 'train':
			full_kspace_dir = kspace_dir.replace(df.iloc[idx]['AF'],'FullSample')
		elif options['mode'] == 'eval':
			full_kspace_dir = kspace_dir
		
		#Load original data
		kspace_dict = loadmat(kspace_dir)
		mask_dict = loadmat(mask_dir)
		kspace_full_dict = loadmat(full_kspace_dir)

		#get variable names from MAT file
		kspace_var = list(kspace_dict.keys())[0]
		mask_var = list(mask_dict.keys())[0]
		kspace_full_var = list(kspace_full_dict.keys())[0]

		#get specified slice and timeframe
		kspace_slice = np.squeeze(kspace_dict[kspace_var][:,:,slice,timeFrame])
		mask = mask_dict[mask_var]
		kspace_full_slice = np.squeeze(kspace_full_dict[kspace_full_var][:,:,slice,timeFrame])

		# compute initial image (Direct IFFT)
		# Kspace >>>>>>IFFT >>>>>>>>> Image
		image_slice = mri_utils.ifft2c(kspace_slice).astype(np.complex64)
		image_slice_full = mri_utils.ifft2c(kspace_full_slice).astype(np.complex64)

		#Zero pad array to 4-level U-net compatiple sizes
		image_slice = pad_array_center(image_slice,(448,224))
		image_slice_full = pad_array_center(image_slice_full,(448,224))

		# normalize the data
		if self.options['normalization'] == 'max':
			norm = np.max(np.abs(image_slice))
		elif self.options['normalization'] == 'no':
			norm = 1.0
		else:
			raise ValueError("Normalization has to be in ['max','no']")

		image_slice /= norm


		input0 = numpy_2_complex(image_slice)
		mask = torch.from_numpy(mask)
		ref = numpy_2_complex(image_slice_full)

		data = {'input0':input0, 'sampling_mask':mask, 'reference':ref}
		return data













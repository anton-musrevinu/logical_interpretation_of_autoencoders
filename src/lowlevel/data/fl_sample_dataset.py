from data.base_dataset import BaseDataset
from util.psdd_interface import decode_binary_to_onehot, decode_binary_to_int,read_info_file,read_info_file_basic
import os.path
import numpy as np
import torch
import scipy.misc

class FLSAMPLEDataset(BaseDataset):
	"""Data provider for MNIST handwritten digit images."""

	def __init__(self, opt, type_of_data, mydir = None):
		"""Initialize this dataset class.

		Parameters:
			opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		BaseDataset.__init__(self, opt)
		if mydir == None:
			raise Exception('Directory has to be supplied')

		self.data_file = os.path.join(mydir, '{}'.format(type_of_data))
		self.type_of_data = type_of_data

		if opt.fl_info_file is not None:
			domains = read_info_file_basic(opt.fl_info_file)
		else:
			domains = read_info_file(self.data_file)

		self.num_classes = opt.num_classes

		fl_data = {}
		self.example_prob = []

		# print('original fl_flat size {} and binary fl size {}'.format(self.model.netAE.fl_flat_shape[1],new_fl_size))

		with open(self.data_file, 'r') as f:
			for line in f:
				if len(line.split(';')) == 2:
					line_prob = float(line.split(';')[1])
					line = line.split(';')[0].split(',')
				else:
					line = line.split(',')
					line_prob = -1
				# print(line)

				for fl_name, fl_part in domains.items():

					if fl_part.bin_encoded:
						flx_var_binary_size = int(np.ceil(np.log2(fl_part.var_cat_dim)))
						flx_elem = np.zeros((fl_part.nb_vars, fl_part.var_cat_dim))
						for new_idx, idx in enumerate(range(fl_part.encoded_start_idx, fl_part.encoded_end_idx, flx_var_binary_size)):
							cat_var_as_bin_list = line[idx:flx_var_binary_size + idx]
							flx_elem[new_idx] = decode_binary_to_onehot(cat_var_as_bin_list, fl_part.var_cat_dim)
					else:
						flx_elem = line[fl_part.encoded_start_idx:fl_part.encoded_end_idx]

					if not fl_name in fl_data:
						fl_data[fl_name] = []
					fl_data[fl_name].append(flx_elem)
				self.example_prob.append(line_prob)

		tmp_fl_name = domains.keys()[0] if fl_name == None else fl_name
		cuccent_size = len(fl_data[tmp_fl_name])
		if cuccent_size < self.batch_size:
			for i in range(cuccent_size, self.batch_size):
				for fl_name, fl_part in domains.items():
					flx_elem = fl_part.get_empty_example()
					fl_data[fl_name].append(flx_elem)

		self.fl_images_names = []
		self.fl_numeric_names = []
		self.fl_data = {}
		for fl_name in fl_data.keys():
			self.fl_data[fl_name] = np.asarray(fl_data[fl_name]).astype(np.float32)
			if 'y' in fl_name:
				self.fl_numeric_names.append(fl_name)
			else:
				self.fl_images_names.append(fl_name)

		print(self)
		# print(np.min(self.inputs), np.max(self.inputs))


		self.num_data_points = int(self.fl_data[self.fl_images_names[0]].shape[0] / self.batch_size) * self.batch_size
		if opt.num_batches != -1:# and trim_data:
			self.num_data_points = min(opt.num_batches * self.batch_size, self.num_data_points)

		# if trim_data:
		# 	self.inputs = np.delete(self.inputs, np.s_[self.num_data_points:], axis = 0)
		# 	self.targets = np.delete(self.targets, np.s_[self.num_data_points:], axis = 0)
	def get_input_shape(self):
		return (self.opt.feature_layer_size, self.opt.categorical_dim)

	def __getitem__(self, index):
		"""Return a data point and its metadata information.

		Parameters:
			index - - a random integer for data indexing

		Returns a dictionary that contains A, B, A_paths and B_paths
			A (tensor) - - the L channel of an image
			B (tensor) - - the ab channels of the same image
			A_paths (str) - - image paths
			B_paths (str) - - image paths (same as A_paths)
		"""
		return_dict = {}
		for i in self.fl_images_names:
			inputs_batch = self.fl_data[i][index].reshape(*self.get_input_shape())
			return_dict[i] = torch.Tensor(inputs_batch).float()
		for i in self.fl_numeric_names:
			targets_batch = self.fl_data[i][index]
			return_dict[i] = torch.Tensor(targets_batch).float()


		return return_dict

	def __len__(self):
		"""Return the total number of images in the dataset."""
		return self.num_data_points

	def __str__(self):
		return ' '.join(['{}.shape: {}'.format(i, self.fl_data[i].shape) for i in self.fl_data.keys()])

	def to_one_of_k(self, int_targets):
		"""Converts integer coded class target to 1 of K coded targets.

		Args:
			int_targets (ndarray): Array of integer coded class targets (i.e.
				where an integer from 0 to `num_classes` - 1 is used to
				indicate which is the correct class). This should be of shape
				(num_data,).

		Returns:
			Array of 1 of K coded targets i.e. an array of shape
			(num_data, num_classes) where for each row all elements are equal
			to zero except for the column corresponding to the correct class
			which is equal to one.
		"""
		one_of_k_targets = np.zeros((self.num_classes))
		one_of_k_targets[int_targets] = 1
		return one_of_k_targets

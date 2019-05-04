from data.base_dataset import BaseDataset
from util.binary_fl import *
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
		self.num_classes = opt.num_classes

		flx_data = []
		fly_data = []
		new_cat_dim = int(np.ceil(np.log2(opt.categorical_dim)))
		new_fl_size = new_cat_dim * opt.feature_layer_size
		# print('original fl_flat size {} and binary fl size {}'.format(self.model.netAE.fl_flat_shape[1],new_fl_size))

		with open(self.data_file, 'r') as f:
			for line in f:
				line = line.split(',')
				flx_elem = np.zeros((opt.feature_layer_size,opt.categorical_dim))
				# print(line)
				for new_idx, idx in enumerate(range(0,new_fl_size,new_cat_dim)):
					cat_var_as_bin_list = line[idx:new_cat_dim + idx]
					flx_elem[new_idx] = decode_binary_to_onehot(cat_var_as_bin_list, opt.categorical_dim)

				fly_elem = decode_binary_to_int(line[new_fl_size:])

				flx_data.append(flx_elem)
				fly_data.append(fly_elem)

		self.flx = np.asarray(flx_data).astype(np.float32)
		self.fly = np.asarray(fly_data).astype(np.float32)

		print(self)
		# print(np.min(self.inputs), np.max(self.inputs))

		self.num_data_points = int(self.flx.shape[0] / self.batch_size) * self.batch_size
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
		inputs_batch = self.flx[index].reshape(*self.get_input_shape())
		# print(index, self.targets[index], type(index), type(self.targets[index]))
		targets_batch = self.to_one_of_k(int(self.fly[index]))

		inputs_batch = torch.Tensor(inputs_batch).float()
		targets_batch = torch.Tensor(targets_batch).float()

		return {'flx': inputs_batch, 'fly': targets_batch, 'indexs': index}

	def __len__(self):
		"""Return the total number of images in the dataset."""
		return self.num_data_points

	def __str__(self):
		return 'flx.shape: {}, fly.shape: {}'.format(self.flx.shape, self.fly.shape)

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

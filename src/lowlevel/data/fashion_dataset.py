from data.base_dataset import BaseDataset
import os.path
import numpy as np
import torch
import scipy.misc

class FASHIONDataset(BaseDataset):
	"""Data provider for FASHIONDataset handwritten digit images."""


	@staticmethod
	def modify_commandline_options(parser, is_train):
		"""Add new dataset-specific options, and rewrite default values for existing options.

		Parameters:
			parser          -- original option parser
			is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.

		By default, the number of channels for input image  is 1 (L) and
		the nubmer of channels for output image is 2 (ab). The direction is from A to B
		"""
		parser.add_argument('--num_classes', type=int, default=10)
		parser.set_defaults(input_nc=1, batch_size=100, image_width = 28, image_height = 28)
		return parser

	def __init__(self, opt, type_of_data, mydir = None):
		"""Initialize this dataset class.

		Parameters:
			opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		BaseDataset.__init__(self, opt)
		if mydir == None:
			self.dir = os.path.join(opt.dataroot, 'fashion-{}.npz'.format(type_of_data))
		else:
			self.dir = os.path.join(mydir, 'fashion-{}.npz'.format(type_of_data))
		self.type_of_data = type_of_data
		self.num_classes = 10

		loaded = np.load(self.dir)
		self.inputs = loaded['inputs'].astype(np.float32)
		if np.max(self.inputs) > 1:
			self.inputs = self.inputs / 255.0
		self.targets = loaded['targets'].astype(np.float32)

		# print(np.min(self.inputs), np.max(self.inputs))

		self.num_data_points = int(self.inputs.shape[0] / self.batch_size) * self.batch_size
		if opt.num_batches != -1:# and trim_data:
			self.num_data_points = min(opt.num_batches * self.batch_size,self.num_data_points)

		# if trim_data:
		# 	self.inputs = np.delete(self.inputs, np.s_[self.num_data_points:], axis = 0)
		# 	self.targets = np.delete(self.targets, np.s_[self.num_data_points:], axis = 0)

	def get_input_shape(self):
		return self.opt.input_nc,self.opt.image_height,self.opt.image_width

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
		inputs_batch = self.inputs[index].reshape(*self.get_input_shape())
		# print(index, self.targets[index], type(index), type(self.targets[index]))
		targets_batch = self.to_one_of_k(int(self.targets[index]))

		inputs_batch = torch.Tensor(inputs_batch).float()
		targets_batch = torch.Tensor(targets_batch).float()

		return {'inputs': inputs_batch, 'targets': targets_batch, 'indexs': index}



	def __len__(self):
		"""Return the total number of images in the dataset."""
		return self.num_data_points

	def __str__(self):
		return 'inputs_shape: {}, target.shape: {}'.format(self.inputs.shape, self.targets.shape)

	def to_one_of_k(self, int_targets, num_classes = None):
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
		if num_classes == None:
			num_classes = self.num_classes
		one_of_k_targets = np.zeros((num_classes))
		one_of_k_targets[int_targets] = 1
		return one_of_k_targets

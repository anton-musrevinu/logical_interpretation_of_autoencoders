from data.base_dataset import BaseDataset
import os.path
import numpy as np
import torch
import scipy.misc

class USPSDataset(BaseDataset):
	"""Data provider for MNIST handwritten digit images."""


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
		parser.set_defaults(input_nc=1, batch_size=100)
		return parser

	def __str__(self):
		return 'inputs_shape: {}, target.shape: {}'.format(self.inputs.shape, self.targets.shape)

	def __init__(self, opt, type_of_data, mydir = None, image_size = None, trim_data = True):
		"""Initialize this dataset class.

		Parameters:
			opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		BaseDataset.__init__(self, opt)
		if mydir == None:
			self.dir = os.path.join(opt.dataroot, 'usps-{}.npz'.format(type_of_data))
		else:
			self.dir = os.path.join(mydir,'usps-{}.npz'.format(type_of_data))

		self.type_of_data = type_of_data
		self.num_classes = 10

		loaded = np.load(self.dir)
		self.inputs = loaded['inputs'].astype(np.float32)
		self.targets = loaded['targets'].astype(np.float32)

		# print(np.min(self.inputs), np.max(self.inputs))

		self.num_data_points = int(self.inputs.shape[0] / self.batch_size) * self.batch_size
		if opt.testing_data and trim_data:
			self.num_data_points = min(5 * self.batch_size,self.num_data_points)

		if trim_data:
			self.inputs = np.delete(self.inputs, np.s_[self.num_data_points:], axis = 0)
			self.targets = np.delete(self.targets, np.s_[self.num_data_points:], axis = 0)

		self.image_num_channels = 1
		self.image_height_org = 16
		self.image_width_org = 16

		if image_size != (self.image_height_org, self.image_width_org) and not 'transfered' in type_of_data:
			self.image_width = image_size[0]
			self.image_height = image_size[1]

			#Version 2: Padding:
			padding = (self.image_width - self.image_width_org) / 2
			if self.image_height == self.image_width and self.image_width_org < self.image_width \
					and padding % 2 == 0:
				padding = int(padding)
				inputs_tmp = np.reshape(self.inputs,(-1,self.image_height_org,self.image_width_org))
				inputs_new = np.zeros((inputs_tmp.shape[0],self.image_height,self.image_width))
				for idx,val in enumerate(inputs_tmp):
					inputs_new[idx] = np.pad(val, padding, 'constant', constant_values = 0)
				print('\t-data- reshaping usps data using {} from {} to {}'.format('padding',inputs_tmp.shape, inputs_new.shape))
				self.inputs = np.reshape(inputs_new,(inputs_new.shape[0], -1))

			else:
				inputs_tmp = np.reshape(self.inputs,(-1,self.image_num_channels,self.image_height_org,self.image_width_org))
				inputs_new = np.zeros((inputs_tmp.shape[0], 1, self.image_height,self.image_width))
				for idx,val in enumerate(inputs_tmp):
					# print(val.shape, val[0].shape)
					tmp_image = scipy.misc.imresize(val[0], size = (self.image_height,self.image_width))
					inputs_new[idx] = np.reshape(tmp_image,(1,*tmp_image.shape))
				print('\t-data- reshaping usps data using {} from {} to {}'.format('scipy.imresize',inputs_tmp.shape, inputs_new.shape))
				self.inputs = np.reshape(inputs_new,(inputs_new.shape[0], -1))

		elif 'transfered' in type_of_data and image_size != None:
			self.image_width = image_size[0]
			self.image_height = image_size[1]
		else:
			self.image_width = self.image_width_org
			self.image_height = self.image_width_org

		# print('\t input: {}, target: {}'.format(self.inputs.shape, self.targets.shape))


	def get_input_shape(self):
		return self.image_num_channels,self.image_height,self.image_width

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
		# batch_slice = slice(index,(index + self.opt.batch_size))

		inputs_batch = self.inputs[index].reshape(*self.get_input_shape())
		targets_batch = self.to_one_of_k(int(self.targets[index]))

		inputs_batch = torch.Tensor(inputs_batch).float()
		targets_batch = torch.Tensor(targets_batch).float()

		return {'inputs': inputs_batch, 'targets': targets_batch, 'indexs': index}

	def __len__(self):
		"""Return the total number of images in the dataset."""
		return self.num_data_points

	def __str__(self):
		return 'inputs_shape: {}, target.shape: {}'.format(self.inputs.shape, self.targets.shape)

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

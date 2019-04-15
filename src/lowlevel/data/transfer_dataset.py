from data.base_dataset import BaseDataset
from data.mnist_dataset import MNISTDataset
from data.usps_dataset import USPSDataset
import os.path
import numpy as np
import torch
import scipy.misc
import random
import importlib


def find_dataset_using_name(dataset_name):
	"""Import the module "data/[dataset_name]_dataset.py".

	In the file, the class called DatasetNameDataset() will
	be instantiated. It has to be a subclass of BaseDataset,
	and it is case-insensitive.
	"""
	dataset_filename = "data." + dataset_name + "_dataset"
	datasetlib = importlib.import_module(dataset_filename)

	dataset = None
	target_dataset_name = dataset_name.replace('_', '') + 'dataset'
	for name, cls in datasetlib.__dict__.items():
		if name.lower() == target_dataset_name.lower() \
		   and issubclass(cls, BaseDataset):
			dataset = cls

	if dataset is None:
		raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

	return dataset

class TransferDataset(BaseDataset):
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


	def __init__(self, opt, type_of_data, mydir = None, image_size = None, trim_data = True):
		"""Initialize this dataset class.

		Parameters:
			opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		BaseDataset.__init__(self, opt)

		dataset_class = find_dataset_using_name('mnist')
		source_data_set = dataset_class(opt, type_of_data, mydir, image_size, trim_data = False)

		dataset_class = find_dataset_using_name('usps')
		target_data_set = dataset_class(opt, type_of_data, mydir, image_size, trim_data = False)

		if opt.allign_data:
			trim_data_points = min(len(source_data_set), len(target_data_set))
			if opt.testing_data and trim_data:
				trim_data_points = min(5 * self.batch_size,trim_data_points)
			#Pair the two datasets, so that len(inputs) == len(targets) and labels_input == labels_target
			inputs, targets, labels_inputs, labels_targets = self.allign_data(source_data_set,target_data_set, trim_data_points)
		else:
			inputs = target_data_set.inputs
			targets = source_data_set.inputs
			labels_inputs = target_data_set.targets
			labels_targets = source_data_set.targets
			trim_data_points = None

		if opt.testing_data and trim_data:
			trim_data_points = min(5 * self.batch_size,len(inputs),len(targets))

		# Inputs = A = Target, target = B = Source
		self.A = np.array(inputs[:trim_data_points])  #USPS data
		self.B = np.array(targets[:trim_data_points]) #MNIST DATA
		self.A_labels = np.array(labels_inputs[:trim_data_points])
		self.B_labels = np.array(labels_targets[:trim_data_points])


		# print('self.A: ',self.A.shape,'   self.B: ',self.B.shape, 'self.A_labels: ', self.A_labels.shape, 'self.B_labels: ', self.B_labels.shape)

		self.image_num_channels = 1
		self.image_height = source_data_set.image_height
		self.image_width = source_data_set.image_width

		# print('\t input: {}, target: {}'.format(self.inputs.shape, self.targets.shape))


	def __str__(self):
		return 'A.shape: {}, B.shape: {}, A_labels.shape: {}, B_labels.shape: {}, allign_data: {}'.format(\
			self.A.shape, self.B.shape,self.A_labels.shape, self.B_labels.shape, self.opt.allign_data)

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

		index_A = index % len(self.A)
		if self.opt.allign_data:   # make sure index is within then range
			index_B = index % len(self.B)
		else:   # randomize the index for domain B to avoid fixed pairs.
			index_B = random.randint(0, len(self.B) - 1)

		A_elem = self.A[index_A].reshape(*self.get_input_shape())
		B_elem = self.B[index_B].reshape(*self.get_input_shape())
		A_labels_elem = self.A_labels[index_A]
		B_labels_elem = self.B_labels[index_B]

		A_elem = torch.Tensor(A_elem).float()
		B_elem = torch.Tensor(B_elem).float()

		return {'A': A_elem, 'B': B_elem, 'A_labels': A_labels_elem, 'B_labels': B_labels_elem }

	def __len__(self):
		"""Return the total number of images in the dataset."""
		return max(len(self.A), len(self.B))

	def allign_data(self,source_data_set, target_data_set, num_data_points):
		if len(source_data_set) < len(target_data_set):
			smaller_data_set = source_data_set
			larger_data_set = target_data_set
		else:
			smaller_data_set = target_data_set
			larger_data_set = source_data_set


		possible_idxs_larger = list(range(len(larger_data_set)))
		possible_idxs_smaller = list(range(len(smaller_data_set)))
		inputs = []
		targets = []
		labels = []

		for i in range(num_data_points):
			smaller_idx_idx = random.choice(range(len(possible_idxs_smaller)))
			smaller_idx = possible_idxs_smaller[smaller_idx_idx]
			smaller_label = smaller_data_set.targets[smaller_idx]
			# print(possible_idxs_smaller, smaller_idx, smaller_label)
			del possible_idxs_smaller[smaller_idx_idx]

			for larger_idx_idx ,larger_idx in enumerate(possible_idxs_larger):
				larger_label = larger_data_set.targets[larger_idx]
				if smaller_label == larger_label:
					break

			if smaller_label != larger_label:
				continue

			del possible_idxs_larger[larger_idx_idx]

			if smaller_data_set == target_data_set:
				inputs.append(smaller_data_set.inputs[smaller_idx])
				targets.append(larger_data_set.inputs[larger_idx])
			else:
				inputs.append(larger_data_set.inputs[larger_idx])
				targets.append(smaller_data_set.inputs[smaller_idx])
			labels.append(smaller_label)
		return inputs, targets, labels, labels

def create_transfered(opt):

	type_of_data = 'test'

	mnist_train = MNISTDataset(opt, 'train', trim_data = False, image_size = (28,28))
	mnist_valid = MNISTDataset(opt, 'valid', trim_data = False, image_size = (28,28))

	mnist_input_data = np.concatenate((mnist_train.inputs, mnist_valid.inputs), axis = 0)
	mnist_target_data = np.concatenate((mnist_train.targets, mnist_valid.targets), axis = 0)

	print(mnist_input_data.shape, mnist_target_data.shape)

	usps_train = USPSDataset(opt, 'train', trim_data = False, image_size = (28,28))
	usps_valid = USPSDataset(opt,'valid',trim_data = False, image_size = (28,28))

	usps_input_data = np.concatenate((usps_train.inputs, usps_valid.inputs), axis = 0)
	usps_target_data = np.concatenate((usps_train.targets, usps_valid.targets), axis = 0)

	print(usps_input_data.shape, usps_input_data.shape)


	return
	possible_idxs_larger = list(range(len(larger_data_set)))
	possible_idxs_smaller = list(range(len(smaller_data_set)))
	inputs = []
	targets = []
	labels = []

	for i in range(len(smaller_data_set)):
		smaller_idx_idx = random.choice(range(len(possible_idxs_smaller)))
		smaller_idx = possible_idxs_smaller[smaller_idx_idx]
		smaller_label = smaller_data_set.targets[smaller_idx]
		# print(possible_idxs_smaller, smaller_idx, smaller_label)
		del possible_idxs_smaller[smaller_idx_idx]

		for larger_idx_idx ,larger_idx in enumerate(possible_idxs_larger):
			larger_label = larger_data_set.targets[larger_idx]
			if smaller_label == larger_label:
				break

		if smaller_label != larger_label:
			continue

		del possible_idxs_larger[larger_idx_idx]


		inputs.append(smaller_data_set.inputs[smaller_idx])
		targets.append(larger_data_set.inputs[larger_idx])
		labels.append(smaller_label)

	usps = np.array(inputs)
	mnist = np.array(targets)
	labels = np.array(labels)

	print('smaller: {}, larger: {}'.format(smaller_data_set, larger_data_set))

	print('usps ', inputs.shape, ' mnist ', mnist.shape, ' label ', labels.shape)

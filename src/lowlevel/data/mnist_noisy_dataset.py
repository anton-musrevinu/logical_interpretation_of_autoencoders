from data.mnist_dataset import MNISTDataset
import os.path
import numpy as np
import random
import torch

class MNISTNOISYDataset(MNISTDataset):
	def __init__(self, opt, type_of_data, mydir = None, noisiness = -1):
		MNISTDataset.__init__(self, opt, type_of_data, mydir)
		if noisiness == -1 or noisiness >= self.num_classes:
			raise Exception('Noisiness -1 is not allowed, please provide a vaid arguement')
		self.noisiness = noisiness

		self.allign_data_for_func()

	def allign_data_for_func(self):

		print('creating noisy-{} dataset.. '.format(self.noisiness))

		new_targets = []
		for i in range(self.num_data_points):
			a_label = int(self.targets[i])
			possible_lables = list(range(self.num_classes))
			del possible_lables[a_label]
			noisy_labels = np.random.choice(possible_lables, self.noisiness)
			new_target = self.to_one_of_k(a_label)
			for j in noisy_labels:
				new_target[j] = 1
			new_targets.append(new_target)
		
		self.targets = np.array(new_targets)

		print('finished ', self)

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
		targets_batch = self.targets[index]

		inputs_batch = torch.Tensor(inputs_batch).float()
		targets_batch = torch.Tensor(targets_batch).float()

		return {'inputs': inputs_batch, 'targets': targets_batch, 'indexs': index}

	def __str__(self):
		return '{} ---- Noisiness: {}'.format(super(MNISTNOISYDataset, self).__str__(), self.noisiness)
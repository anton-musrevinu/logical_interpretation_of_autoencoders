from data.mnist_dataset import MNISTDataset
import os.path
import numpy as np
import random

class MNISTG7LANDDataset(MNISTDataset):
	def __init__(self, opt, type_of_data, mydir = None):
		MNISTDataset.__init__(self, opt, type_of_data, mydir)

		self.allign_data_for_func()

	def allign_data_for_func(self, relational_func = lambda a,b: a > 7 and b > 7 ,  add_y = False):

		possible_idxs = list(range(len(self.inputs)))
		domain_a = []
		domain_b = []
		y_label = []

		print('alligning data set.. ')

		last_label = None
		for i in range(self.num_data_points):
			a_idx_idx = random.choice(range(len(possible_idxs)))
			a_idx = possible_idxs[a_idx_idx]
			a_label = self.targets[a_idx]
			# print(possible_idxs_smaller, smaller_idx, smaller_label)
			del possible_idxs[a_idx_idx]

			if len(domain_a) <= len(domain_b):
				domain_a.append(a_idx)
				last_label = a_label
			else:
				domain_b.append(a_idx)
				y_label.append(relational_func(last_label, a_label))


		self.num_data_points = int(len(y_label) / self.batch_size) * self.batch_size
		
		self.domain_a = domain_a
		self.domain_b = domain_b
		self.y_label = y_label

		print('finished ', self)


	def __str__(self):
		return 'domains: 2, num_points: {}, domain_x {}'.format(self.num_data_points, super(MNISTG7LANDDataset, self).__str__())

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

		return {'domain_a': super(MNISTG7LANDDataset, self).__getitem__(self.domain_a[index]),\
				 'domain_b': super(MNISTG7LANDDataset, self).__getitem__(self.domain_b[index]), \
				 'y_label': self.to_one_of_k(int(self.y_label[index]), 2)}
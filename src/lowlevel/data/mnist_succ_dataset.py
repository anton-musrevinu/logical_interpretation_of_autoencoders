from data.mnist_dataset import MNISTDataset
import os.path
import numpy as np
import random

class MNISTSUCCDataset(MNISTDataset):
	def __init__(self, opt, type_of_data, mydir = None):
		MNISTDataset.__init__(self, opt, type_of_data, mydir)

		self.allign_data_for_func()

	def allign_data_for_func(self, relational_func = lambda a,b: a == b - 1 ,  add_y = False):

		possible_idxs = list(range(len(self.inputs)))
		domain_a = []
		domain_b = []

		print('alligning data set.. ')

		for i in range(self.num_data_points):
			a_idx_idx = random.choice(range(len(possible_idxs)))
			a_idx = possible_idxs[a_idx_idx]
			a_label = self.targets[a_idx]
			# print(possible_idxs_smaller, smaller_idx, smaller_label)
			del possible_idxs[a_idx_idx]

			for b_idx_idx, b_idx in enumerate(possible_idxs):
				b_label = self.targets[b_idx]
				if relational_func(a_label, b_label):
					break

			if not relational_func(a_label, b_label):
				continue

			del possible_idxs[b_idx_idx]

			domain_a.append(a_idx)
			domain_b.append(b_idx)

			if len(possible_idxs) <= 100:
				break

		self.num_data_points = len(domain_a)
		
		self.domain_a = domain_a
		self.domain_b = domain_b

		print('finished ', self)


	def __str__(self):
		return 'domains: 2, num_points: {}, domain_x {}'.format(self.num_data_points, super(MNISTSUCCDataset, self).__str__())

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

		return {'domain_a': super(MNISTSUCCDataset, self).__getitem__(self.domain_a[index]),\
				 'domain_b': super(MNISTSUCCDataset, self).__getitem__(self.domain_b[index])}
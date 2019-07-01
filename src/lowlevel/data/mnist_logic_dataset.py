from data.mnist_dataset import MNISTDataset
import os.path
import numpy as np
import random

class MNISTLOGICDataset(MNISTDataset):
	def __init__(self, opt, type_of_data, mydir = None,\
		 additional_constraint_on_data = lambda a,b: True, \
		 relational_func = lambda a,b: a and b,\
		  domain_constraints = lambda a: True,
		  y_classes = 2):
		MNISTDataset.__init__(self, opt, type_of_data, mydir)
		self.additional_constraint_on_data = additional_constraint_on_data
		self.relational_func = relational_func
		self.domain_constraints = domain_constraints
		self.y_classes = y_classes

		self.allign_data_for_func(additional_constraint_on_data, relational_func, domain_constraints)

	def allign_data_for_func(self, additional_constraint_on_data, relational_func, domain_constraints):

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

			if domain_constraints(a_label):
				if len(domain_a) <= len(domain_b):
					if additional_constraint_on_data(a_label, 'domain_a'):
						domain_a.append(a_idx)
						last_label = a_label
				else:
					if additional_constraint_on_data(a_label, 'domain_b'):
						domain_b.append(a_idx)
						y_label.append(relational_func(last_label, a_label))


		self.num_data_points = int(len(y_label) / self.batch_size) * self.batch_size
		
		self.domain_a = domain_a
		self.domain_b = domain_b
		self.y_label = y_label

		print('finished ', self)


	def __str__(self):
		return 'domains: 2, num_points: {}, domain_x {}'.format(self.num_data_points, super(MNISTLOGICDataset, self).__str__())

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

		return {'domain_a': super(MNISTLOGICDataset, self).__getitem__(self.domain_a[index]),\
				 'domain_b': super(MNISTLOGICDataset, self).__getitem__(self.domain_b[index]), \
				 'y_label': self.to_one_of_k(int(self.y_label[index]), self.y_classes)}
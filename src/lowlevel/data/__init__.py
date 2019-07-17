"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
	-- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
	-- <__len__>:                       return the size of dataset.
	-- <__getitem__>:                   get a data point from data loader.
	-- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import scipy.misc
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


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


def get_option_setter(dataset_name):
	"""Return the static method <modify_commandline_options> of the dataset class."""
	dataset_class = find_dataset_using_name(dataset_name)
	return dataset_class.modify_commandline_options

def create_dataset_new(opt, domain, type_of_data, mydir = None, args_for_dataset = None):
	data_loader = CustomDatasetDataLoader(opt, domain, type_of_data, mydir, args_for_dataset)
	dataset = data_loader.load_data()
	return dataset

def create_transfer_dataset(opt, domainA, domainB, type_of_data,mydir = None, args_for_dataset = None):
	data_loader = CustomTransferDatasetDataLoader(opt, domainA, domainB, type_of_data, mydir, args_for_dataset)
	dataset = data_loader.load_data()
	return dataset

class CustomTransferDatasetDataLoader():
	"""Wrapper class of Dataset class that performs multi-threaded data loading"""

	def __init__(self, opt, domainA, domainB, type_of_data, mydir, args_for_dataset = None):
		"""Initialize this class

		Step 1: create a dataset instance given the name [dataset_mode]
		Step 2: create a multi-threaded data loader.
		"""
		self.opt = opt
		datasetA_class = find_dataset_using_name(domainA)
		if args_for_dataset != None:
			datasetA = dataset_class(opt, type_of_data, mydir, **args_for_dataset)
		else:
			datasetA = dataset_class(opt, type_of_data, mydir)
		self.dataloader = torch.utils.data.DataLoader(
			dataset,
			batch_size=opt.batch_size,
			shuffle=True,#not opt.allign_data,
			num_workers=int(opt.num_threads))

		asStr = '{}-{}'.format(domain, type_of_data)
		self.dataset = dataset
		self.str = 'dataset-{}-{}'.format(asStr, dataset)


		print("[{} - {}] Id: {} , batch_size: {}, num_batches: {}".format(\
			type(dataset).__name__,type_of_data,self.str,opt.batch_size,len(self)))

	# def get_num_batches(self):
	# 	return len(self.dataloader)

	def __getitem__(self,index):
		return self.dataset[index]

	def load_data(self):
		return self

	def __len__(self):
		"""Return the number of data in the dataset"""
		return len(self.dataloader)

	def __str__(self):
		return self.str

	def __iter__(self):
		"""Return a batch of data"""
		for i, data in enumerate(self.dataloader):
				# print(i, data['inputs'].shape,data['targets'].shape, type(data))
				yield data

# def create_dataset(opt, type_of_data = 'train', mytype = None ):
# 	"""Create a dataset given the option.

# 	This function wraps the class CustomDatasetDataLoader.
# 		This is the main interface between this package and 'train.py'/'test.py'

# 	Example:
# 		>>> from data import create_dataset
# 		>>> dataset = create_dataset(opt)
# 	"""
# 	if mytype == 'transfer':
# 		data_loader = MyCustomDatasetDataLoader(opt, type_of_data)
# 	else:
# 		data_loader = CustomDatasetDataLoader(opt, type_of_data)

# 	if opt.testing:
# 		data_loader.num_batches = 5
# 	dataset = data_loader.load_data()
# 	return dataset

class CustomDatasetDataLoader():
	"""Wrapper class of Dataset class that performs multi-threaded data loading"""

	def __init__(self, opt, domain, type_of_data, mydir, args_for_dataset = None):
		"""Initialize this class

		Step 1: create a dataset instance given the name [dataset_mode]
		Step 2: create a multi-threaded data loader.
		"""
		self.opt = opt
		dataset_class = find_dataset_using_name(domain)
		if args_for_dataset != None:
			dataset = dataset_class(opt, type_of_data, mydir, **args_for_dataset)
		else:
			dataset = dataset_class(opt, type_of_data, mydir)
		self.dataloader = torch.utils.data.DataLoader(
			dataset,
			batch_size=opt.batch_size,
			shuffle=True,#not opt.allign_data,
			num_workers=int(opt.num_threads))

		asStr = '{}-{}'.format(domain, type_of_data)
		self.dataset = dataset
		self.str = 'dataset-{}-{}'.format(asStr, dataset)


		print("[{} - {}] Id: {} , batch_size: {}, num_batches: {}".format(\
			type(dataset).__name__,type_of_data,self.str,opt.batch_size,len(self)))

	# def get_num_batches(self):
	# 	return len(self.dataloader)

	def __getitem__(self,index):
		return self.dataset[index]

	def load_data(self):
		return self

	def __len__(self):
		"""Return the number of data in the dataset"""
		return len(self.dataloader)

	def __str__(self):
		return self.str

	def __iter__(self):
		"""Return a batch of data"""
		for i, data in enumerate(self.dataloader):
				# print(i, data['inputs'].shape,data['targets'].shape, type(data))
				yield data

		# for i in range(self.num_batches):
		# 	yield self.dataset[i]


class MyCustomDatasetDataLoader():
	"""Wrapper class of Dataset class that performs multi-threaded data loading"""

	def __init__(self, opt, type_of_data = 'train'):
		"""Initialize this class

		Step 1: create a dataset instance given the name [dataset_mode]
		Step 2: create a multi-threaded data loader.
		"""
		self.opt = opt
		dataset_class_source = find_dataset_using_name(opt.source_data)
		dataset_class_target = find_dataset_using_name(opt.target_data)
		self.dataset_source = dataset_class_source(opt, type_of_data)
		self.dataset_target = dataset_class_target(opt, type_of_data)
		print("source dataset [%s] was created" % type(self.dataset_source).__name__)
		print("target dataset [%s] was created" % type(self.dataset_target).__name__)

		self.num_batches = int(len(self) / self.opt.batch_size)

			#not the right kind of subclass

	def load_data(self):
		return self

	def __len__(self):
		"""Return the number of data in the dataset"""
		return min(len(self.dataset_target),len(self.dataset_source), self.opt.max_dataset_size)

	def __iter__(self):
		"""Return a batch of data"""

		if True:
			for i in range(self.num_batches):
				A = torch.Tensor(self.dataset_target[i]['inputs']).float()
				B = torch.Tensor(self.dataset_source[i]['inputs']).float()
				A = A.view(-1, *self.dataset_target.get_input_shape())
				B = B.view(-1, *self.dataset_source.get_input_shape())
				# if target_shape != None:

				# print(A.shape, B.shape)

				yield {'A': A, 'B': B}
		else:
			for i, data in enumerate(self.dataloader):
				if i * self.opt.batch_size >= self.opt.max_dataset_size:
					break
				yield data


from .base_manager import BaseManager
from options import str2bool
from models import create_model
import torch
import numpy as np
import os
from util.storage_utils import save_example_image, save_feature_layer_example
from data import create_dataset_new as create_dataset
from util.psdd_interface import write_fl_batch_to_file,write_fl_batch_to_file_new, FlDomainInfo
import tqdm

class MSCManager(BaseManager):

	def __init__(self,opt):


		self.train_data = create_dataset(opt, opt.dataset, 'train')
		self.val_data = create_dataset(opt, opt.dataset, 'valid')

		BaseManager.__init__(self,opt)

		self.experiment_saved_data = os.path.join(self.manager_dir, "../encoded_data")

	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		"""Add new dataset-specific options, and rewrite default values for existing options.

		Parameters:
			parser          -- original option parser
			is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.
		"""

		# parser.add_argument('--model_type', nargs="?", type=str, default='vae', help='Type of Autoencoder used: [base,lin, vae]')
		parser.add_argument('--feature_layer_size', nargs="?",type=int, default=64)
		parser.add_argument('--categorical_dim', nargs="?",type=int, default=4)
		# parser.add_argument('--use_bias', nargs="?",type=str2bool, default=True)
		#                     help='Experiment name - to be used for building the experiment folder')
		# parser.add_argument('--replace_existing', nargs="?",type=self.str2bool, default=False,
		#                     help='Specify if an exerpiment directory with the same name should be overwritten or not')
		# parser.add_argument('--load', nargs="?", type=str, default="None",
		#                         help='Experiment folder to loads')

		return parser

	def _save_example_batch(self,epoch_idx):
		# print(self.model.feature_layer.shape)

		fl_as_img = self.model.get_fl_as_img()

		# print(fl_as_img.shape, self.model.input.shape, self.model.rec_input.shape)

		save_stuff = [self.model.input, fl_as_img, self.model.rec_input]
		row = map(lambda img_batch: img_batch.cpu().float(), save_stuff)
		row_np = torch.cat(list(row), 3)[:21]
		path = os.path.join(self.experiment_logs, 'transfer_example_epoch_{}.png'.format(epoch_idx))
		# print(row_np.shape, row_np.max(), row_np.min())
		save_example_image(row_np,path, nrow = 3)

	def create_impossible_test_set(self, task_type):
		#Create a dataset for the 'land' tasks, where fly is one and the one image given is 0 (impossible)
		if task_type not in ['bland', 'g7land', 'g4land', 'blor']:
			return

		if not os.path.exists(self.opt.encoded_data_dir):
			os.mkdir(self.opt.encoded_data_dir)

		type_of_data = 'test'

			# not (domain_x == 'domain_b' and x_label == 1)
		relational_func = lambda a, b: True
		domain_constraints = lambda label: True
		if task_type == 'bland':
			additional_constraint_on_data = lambda x_label,domain_x: not (domain_x == 'domain_b' and x_label == 1)
			domain_constraints = lambda label: label == 0 or label == 1
		elif task_type == 'blor':
			additional_constraint_on_data = lambda x_label,domain_x: not (domain_x == 'domain_b' and x_label == 0)
			domain_constraints = lambda label: label == 0 or label == 1

		elif task_type   == 'g7land':
			additional_constraint_on_data = lambda x_label,domain_x: not (domain_x == 'domain_b' and x_label > 7)

		elif task_type == 'g4land':
			additional_constraint_on_data = lambda x_label,domain_x: not (domain_x == 'domain_b' and x_label > 4)

		args_for_dataset = {'additional_constraint_on_data': additional_constraint_on_data, \
							'relational_func': relational_func,\
							'domain_constraints': domain_constraints}

		file_encoded_path = os.path.join(self.opt.encoded_data_dir,'{}_{}-encoded-{}_impossible.data'.format(self.opt.dataset, task_type, type_of_data))
		self.encode_logic_dataset(file_encoded_path,task_type,  type_of_data, self.opt.limit_conversion, args_for_dataset = args_for_dataset)

	def decode_specific_file(self, file_to_decode, output_image_file = None):

		sampled_data = create_dataset(self.opt, domain = 'fl_sample', type_of_data = file_to_decode.split('/')[-1], \
			mydir = '/'.join(file_to_decode.split('/')[:-1]))

		generated_fls = file_to_decode.split('/')[-1].split('generated_')[1].split('-')[0].split('_')
		print('generated fls: {}'.format(generated_fls))
		self.load_net_at_best_epoch()

		for idx, data in enumerate(sampled_data):  # sample batch
			rec = {}
			for i in data.keys():
				if not 'y' in i:
					self.model.set_fl(data[i])
					self.model.run_decoder()
					rec[i] = self.model.rec_input.detach().cpu().float()

			if output_image_file == None:
				path = file_to_decode.replace('.data', 'b{}.png'.format(idx))
			else:
				path = output_image_file

			for i in generated_fls:
				for image in rec[i]:
					image[:,0,:] = 0.5
					image[:,-1,:] = 0.5
					image[:,:,0] = 0.5
					image[:,:,-1] = 0.5

			tosave = list(rec.values())
			save_example_image(tosave, path)

	def decode_specific_file_anverage(self, file_to_decode, output_image_file = None):

		sampled_data = create_dataset(self.opt, domain = 'fl_sample', type_of_data = file_to_decode.split('/')[-1], \
			mydir = '/'.join(file_to_decode.split('/')[:-1]))

		self.load_net_at_best_epoch()

		average_images = []
		for idx, data in enumerate(sampled_data):  # sample batch
			rec = {}
			for i in data.keys():
				if not 'y' in i:
					self.model.set_fl(data[i])
					self.model.run_decoder()
					rec[i] = self.model.rec_input.detach().cpu().float()

			if len(rec.keys()) > 1:
				raise Exception("Multiple fls are not supported yet")
			i = list(rec.keys())[0]
			images = rec[i]

			for image in images:
				image[:,0,:] = 0.5
				image[:,-1,:] = 0.5
				image[:,:,0] = 0.5
				image[:,:,-1] = 0.5

			average_images.append(torch.mean(images, dim = 0, keepdim = True))


		if output_image_file == None:
			path = file_to_decode.replace('.data', '.png')
		else:
			path = output_image_file

		save_example_image(average_images, path)


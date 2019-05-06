
from .base_manager import BaseManager
from options import str2bool
from models import create_model
import torch
import numpy as np
import os
from util.storage_utils import save_example_image, save_feature_layer_example
from data import create_dataset_new as create_dataset
from util.psdd_interface import write_fl_batch_to_file
import tqdm

class VAEManager(BaseManager):

	def __init__(self,opt):


		self.train_data = create_dataset(opt, opt.dataset, 'train')
		self.val_data = create_dataset(opt, opt.dataset, 'valid')


		self.annealing_temp = 1
		self.annealing_temp_min = .5
		self.annealing_rate = 0.00003

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
		parser.add_argument('--use_bias', nargs="?",type=str2bool, default=True)
		parser.add_argument('--num_channels', nargs="?",type=int, default=64)
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


	def record_additional_info(self,epoch_idx):
		if epoch_idx % self.opt.save_epoch_freq == 0:
			self._save_example_batch(epoch_idx)
		if epoch_idx == 1:
			# save model and best val idx and best val acc, using the model dir, model name and model idx
			save_feature_layer_example(model_save_dir=self.experiment_logs,
					model_save_name="feature_layer_eval", model_idx=epoch_idx, 
					feature_layer_ex = self.model.feature_layer[0], 
					feature_layer_hidden_ex = self.model.feature_layer_prob[0])


	def after_training_updates(self,epoch_idx, batch_idx, losses):
		if batch_idx % 100 == 1:
			annealing_temp_old = self.annealing_temp
			new_temp = self.annealing_temp * np.exp(-self.annealing_rate * batch_idx)
			self.annealing_temp = np.maximum(new_temp, self.annealing_temp_min)
			self.model.annealing_temp = self.annealing_temp
			# print('annealing_temp has been updated from: {} to: {}'.format(annealing_temp_old,self.annealing_temp))
		if epoch_idx == 1 and batch_idx == 1:
			save_feature_layer_example(model_save_dir=self.experiment_logs,
					model_save_name="feature_layer_training", model_idx=epoch_idx, 
					feature_layer_ex = self.model.feature_layer[0], 
					feature_layer_hidden_ex = self.model.feature_layer_prob[0])

	def convert_all_data(self):

		if not os.path.exists(self.experiment_saved_data):
			os.mkdir(self.experiment_saved_data)

		for type_of_data in ['train', 'valid', 'test']:
			file_encoded_path = os.path.join(self.experiment_saved_data,'{}-encoded-{}.data'.format(self.opt.dataset, type_of_data))
			self.encode_specific_file(file_encoded_path, type_of_data, self.opt.limit_conversion, self.opt.compress_fly)

	def encode_specific_file(self, file_encoded_path, type_of_data = 'train', limit_conversion = -1, compress_fly = True):
		#Create specified dataset
		dataset_to_encode = create_dataset(self.opt, self.opt.dataset, type_of_data = type_of_data)

		# Set up network with the screenshot from the best performing batch
		key = 'valid_{}'.format(self.opt.for_error.upper())
		if not key in self.best_val_model_idx:
			raise Exception('The network does not hold information for the provided error: {} (key: {})'.format(for_error, key))
		epoch_idx = self.best_val_model_idx[key]
		self.model.load_networks(epoch_idx)
		self.model.annealing_temp = self.annealing_temp_min

		# cat_dim = self.opt.categorical_dim
		# fl_cat_size = self.opt.feature_layer_size
		flx_compressed_var_length = int(np.ceil(np.log2(self.opt.categorical_dim)))
		flx_compressed_size = flx_compressed_var_length * self.opt.feature_layer_size
		if compress_fly:
			fly_size = int(np.ceil(np.log2(dataset_to_encode.dataset.num_classes)))
		else:
			fly_size = dataset_to_encode.dataset.num_classes

		stored_elements = 0
		total_wrt_batch = len(dataset_to_encode)

		if limit_conversion != -1:
			total_wrt_batch = min(total_wrt_batch, int(np.ceil(limit_conversion/self.opt.batch_size)))

		with tqdm.tqdm(total=total_wrt_batch) as pbar:
			for batch_idx, data in enumerate(dataset_to_encode):  # sample batch

				self.model.set_input(data)
				self.model.run_encoder()

				# print(data['targets'].shape, self.model.feature_layer.shape)

				flx_categorical = self.model.feature_layer.detach().numpy() #.view(self.model.feature_layer.shape[0], -1)
				fly_onehot = data['targets'].detach().numpy()
				fl_encoded_size = write_fl_batch_to_file(file_encoded_path, flx_categorical, fly_onehot, batch_idx, compress_fly = compress_fly)
				pbar.update(1)

				stored_elements += self.opt.batch_size
				if limit_conversion != -1 and stored_elements >= limit_conversion:
					break

		print('[ENCODE]\t finished converting dataset: {} - size: ({},{}) \n\t\t to file: {}'.format(dataset_to_encode, stored_elements, fl_encoded_size,file_encoded_path))
		return True

	def decode_specific_file(self, file_to_decode, output_image_file = None):
		key = 'valid_{}'.format(self.opt.for_error.upper())
		if not key in self.best_val_model_idx:
			raise Exception('The network does not hold information for the provided error: {} (key: {})'.format(for_error, key))

		# print(file_to_decode.split('/')[-1],'/'.join(file_to_decode.split('/')[:-1]))
		sampled_data = create_dataset(self.opt, domain = 'fl_sample', type_of_data = file_to_decode.split('/')[-1], \
			mydir = '/'.join(file_to_decode.split('/')[:-1]))

		epoch_idx = self.best_val_model_idx[key]
		#Load the best model for the given key
		self.model.load_networks(epoch_idx)
		self.model.annealing_temp = self.annealing_temp_min

		for idx, data in enumerate(sampled_data):  # sample batch
			self.model.set_fl(data)
			self.model.run_decoder()

			if output_image_file == None:
				path = file_to_decode.replace('.data', 'b{}.png'.format(idx))
			else:
				path = output_image_file
			save_example_image(self.model.rec_input.cpu().float(), path)

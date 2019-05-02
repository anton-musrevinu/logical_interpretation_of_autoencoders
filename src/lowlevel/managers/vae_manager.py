
from .base_manager import BaseManager
from options import str2bool
from models import create_model
import torch
import numpy as np
import os
from util.storage_utils import save_example_image, save_feature_layer_example
from data import create_dataset_new as create_dataset
from util.psdd_interface import write_batch_to_file
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

		test_data = create_dataset(self.opt, self.opt.dataset, 'test')

		self._encode_dataset(self.train_data,self.opt.for_error, self.opt.limit_conversion)
		self._encode_dataset(self.val_data,self.opt.for_error,self.opt.limit_conversion)
		self._encode_dataset(test_data,self.opt.for_error,self.opt.limit_conversion)

	def _encode_dataset(self, dataSet, for_error, limit_conversion = -1, compress = True):
		transfer_result = {}
		key = 'valid_{}'.format(for_error.upper())
		if not key in self.best_val_model_idx:
			raise Exception('The network does not hold information for the provided error: {} (key: {})'.format(for_error, key))

		epoch_idx = self.best_val_model_idx[key]
		#Load the best model for the given key
		self.model.load_networks(epoch_idx)
		self.model.annealing_temp = self.annealing_temp_min

		batches_inputs = []
		batches_targets = []
		batches_input_idx = []
		data_id = '{}-encoded-{}-{}'.format(self.opt.dataset, key, dataSet.dataset.type_of_data)
		con_val_data_save = os.path.join(self.experiment_saved_data,'{}.data'.format(data_id))

		cat_dim = self.opt.categorical_dim
		fl_cat_size = self.opt.feature_layer_size
		if compress:
			new_cat_dim = int(np.ceil(np.log2(cat_dim)))
			new_fl_size = new_cat_dim * fl_cat_size

			fly_size = int(np.ceil(np.log2(dataSet.dataset.num_classes)))

		stored_size = 0
		with tqdm.tqdm(total=len(dataSet)) as pbar:
			for idx_data, data in enumerate(dataSet):  # sample batch

				self.model.set_input(data)
				self.model.run_encoder()

				# print(data['targets'].shape, self.model.feature_layer.shape)

				feature_layer_flat = self.model.feature_layer.clone().view(self.model.feature_layer.shape[0], -1)
				if compress:
					new_feature_layer_flat = torch.zeros(feature_layer_flat.shape[0], new_fl_size)
					# print('original fl_flat: {}'.format(feature_layer_flat.shape))
					# print('new      fl_flat: {}'.format(new_feature_layer_flat.shape))
					for batch_idx, line in enumerate(feature_layer_flat):
						for new_idx, idx in enumerate(range(0,len(line),self.opt.categorical_dim)):
							new_idx = new_idx * new_cat_dim
							cat_value = torch.argmax(line[idx:idx+self.opt.categorical_dim])
							binary_encoded_var = torch.IntTensor(list(map(int,np.binary_repr(cat_value,new_cat_dim))))
							new_feature_layer_flat[batch_idx,new_idx:new_idx + new_cat_dim] = binary_encoded_var
						# print('storing {}[{},{}] ({}) as {}[{},{}]'.format(line[idx:idx+4], idx, idx+3, cat_value, new_feature_layer_flat[batch_idx,new_idx:new_idx + 2], new_idx, new_idx + 1))
					flx = new_feature_layer_flat

					labels_as_bin = torch.zeros(feature_layer_flat.shape[0], fly_size)
					for idx_label, line in enumerate(data['targets']):
						cat_value_bin = np.binary_repr(torch.argmax(line),fly_size)
						binary_list = torch.IntTensor(list(map(int,cat_value_bin)))
						labels_as_bin[idx_label] = binary_list

					fly = labels_as_bin
				else:
					flx = feature_layer_flat
					fly = data['targets']
					# print('converting {} -> {} and storing at: {}'.format(line, binary_list, idx_label))

				# print(labels_as_bin)
				batch_data = torch.cat((flx,fly),1)
				first_line = 'flx: {}, fly: {}, fl_var_dim: {}'.format(flx.shape, fly.shape, cat_dim)
				if idx_data == 1:
					print(first_line)
				write_batch_to_file(con_val_data_save, batch_data, not idx_data == 0, first_line)

				stored_size += self.opt.batch_size
				if limit_conversion != -1 and stored_size > limit_conversion:
					break
				pbar.update(1)

		print('-- finished converting dataset: {} to {} - size: {}'.format(dataSet, data_id,stored_size))
		return True

	def decode_specific_file(self, file_to_decode):
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
			path = file_to_decode.replace('.data', 'b{}.png'.format(idx))
			save_example_image(self.model.rec_input.cpu().float(), path)

		# num_batches = -1
		# flx_data = []
		# fly_data = []
		# new_cat_dim = int(np.ceil(np.log2(self.opt.categorical_dim)))
		# new_fl_size = new_cat_dim * self.opt.feature_layer_size
		# print('original fl_flat size {} and binary fl size {}'.format(self.model.netAE.fl_flat_shape[1],new_fl_size))

		# with open(file_to_decode, 'r') as f:
		# 	for line in f:
		# 		line = line.split(',')
		# 		flx_elem = np.zeros((self.opt.feature_layer_size,self.opt.categorical_dim))
		# 		# print(line)
		# 		for new_idx, idx in enumerate(range(0,new_fl_size,new_cat_dim)):
		# 			cat_var_as_bin_list = line[idx:new_cat_dim + idx]
		# 			flx_elem[new_idx] = self._decode_binary_to_onehot(cat_var_as_bin_list, self.opt.categorical_dim)

		# 		fly_elem = self._decode_binary_to_int(line[new_fl_size:])

		# 		flx_data.append(flx_elem)
		# 		fly_data.append(fly_elem)

		# num_elements_found = len(flx_data)
		# flx_tensor = torch.FloatTensor(flx_data)
		# fly_tensor = torch.FloatTensor(fly_data)

		# for idx in range(num_elements_found// opt.batch_size)



	def decode_dataset(self, dataSet):
		transfer_result = {}
		for key, best_val_model_idx_for_loss in self.best_val_model_idx.items():
			#Load the best model for the given key
			self.model.load_networks(best_val_model_idx_for_loss)

			batches_inputs = []
			batches_targets = []
			batches_input_idx = []
			data_id = '{}-encoded-{}'.format(dataSet.dataset.type_of_data, key)
			con_val_data_save = os.path.join(self.experiment_logs,'{}-{}.npz'.format(opt.dataset,data_id))

			for idx, data in enumerate(dataSet):  # sample batch
				feature_layer = data['feature_layer']

				output_images = self.run_decoder(feature_layer=feature_layer)
				write_batch_to_file(con_val_data_save, output_images, not idx == 0)

		return 

	# def create_model(self):
	# 	# print(self.opt.input_nc, self.opt.output_nc)
	# 	self.opt.model = 'cycle_gan'
	# 	self.model = create_model(self.opt)
	# 	self.model.setup(self.opt)  # re-initialize network parameters
	# 	self.model.save_dir = self.experiment_saved_models

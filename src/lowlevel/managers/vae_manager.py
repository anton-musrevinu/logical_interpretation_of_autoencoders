
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

	def convert_all_data(self, task_type):

		if not os.path.exists(self.opt.encoded_data_dir):
			os.mkdir(self.opt.encoded_data_dir)

		if task_type == 'classification':
			for type_of_data in ['train', 'valid', 'test']:
				file_encoded_path = os.path.join(self.opt.encoded_data_dir,'{}-encoded-{}.data'.format(self.opt.dataset, type_of_data))
				dataset_to_encode = create_dataset(self.opt, self.opt.dataset, type_of_data = type_of_data)
				
				self.encode_2_part_dataset(file_encoded_path, dataset_to_encode, self.opt.limit_conversion, y_classes = dataset_to_encode.dataset.num_classes,\
					compress_fly = self.opt.compress_fly)
		elif task_type == 'successor':
			for type_of_data in ['train', 'valid', 'test']:
				file_encoded_path = os.path.join(self.opt.encoded_data_dir,'{}_succ-encoded-{}.data'.format(self.opt.dataset, type_of_data))
				self.encode_successor_dataset(file_encoded_path, type_of_data, self.opt.limit_conversion)
		elif task_type == 'bland':
			for type_of_data in ['train', 'valid', 'test']:
				file_encoded_path = os.path.join(self.opt.encoded_data_dir,'{}_bland-encoded-{}.data'.format(self.opt.dataset, type_of_data))
				args_for_dataset = {'relational_func'   : lambda a,b: a and b, \
									'domain_constraints': lambda label: label == 0 or label == 1}
				self.encode_logic_dataset(file_encoded_path,task_type,  type_of_data, self.opt.limit_conversion, args_for_dataset)
		elif task_type == 'blor':
			for type_of_data in ['train', 'valid', 'test']:
				file_encoded_path = os.path.join(self.opt.encoded_data_dir,'{}_blor-encoded-{}.data'.format(self.opt.dataset, type_of_data))
				args_for_dataset = {'relational_func'   : lambda a,b: a or b, \
									'domain_constraints': lambda label: label == 0 or label == 1}
				self.encode_logic_dataset(file_encoded_path,task_type,  type_of_data, self.opt.limit_conversion, args_for_dataset)
		elif task_type == 'blxor':
			for type_of_data in ['train', 'valid', 'test']:
				file_encoded_path = os.path.join(self.opt.encoded_data_dir,'{}_blxor-encoded-{}.data'.format(self.opt.dataset, type_of_data))
				args_for_dataset = {'relational_func':    lambda a,b: a != b, \
									'domain_constraints': lambda label: label == 0 or label == 1}
				self.encode_logic_dataset(file_encoded_path,task_type,  type_of_data, self.opt.limit_conversion, args_for_dataset)
		elif task_type == 'g4land':
			for type_of_data in ['train', 'valid', 'test']:
				file_encoded_path = os.path.join(self.opt.encoded_data_dir,'{}_g4and-encoded-{}.data'.format(self.opt.dataset, type_of_data))
				args_for_dataset = {'relational_func': lambda a,b: (a > 4) and (b > 4)}
				self.encode_logic_dataset(file_encoded_path,task_type,  type_of_data, self.opt.limit_conversion, args_for_dataset)
		elif task_type == 'g7land':
			for type_of_data in ['train', 'valid', 'test']:
				file_encoded_path = os.path.join(self.opt.encoded_data_dir,'{}_g7and-encoded-{}.data'.format(self.opt.dataset, type_of_data))
				args_for_dataset = {'relational_func': lambda a,b: (a > 7) and (b > 7)}
				self.encode_logic_dataset(file_encoded_path,task_type,  type_of_data, self.opt.limit_conversion, args_for_dataset)
		elif task_type == 'plus':
			for type_of_data in ['train', 'valid', 'test']:
				file_encoded_path = os.path.join(self.opt.encoded_data_dir,'{}_plus-encoded-{}.data'.format(self.opt.dataset, type_of_data))
				args_for_dataset = {'relational_func': lambda a,b: a + b}
				self.encode_logic_dataset(file_encoded_path,task_type, type_of_data, self.opt.limit_conversion, y_classes = 19)
		elif task_type.startswith('noisy-'):
			noisiness = int(task_type.split('-')[1])
			args_for_dataset = {'noisiness': noisiness}
			for type_of_data in ['train', 'valid']:

				#Noisy dataset for psdd training
				file_encoded_path = os.path.join(self.opt.encoded_data_dir,'{}_{}-encoded-{}.data'.format(self.opt.dataset, task_type ,type_of_data))
				dataset_to_encode = create_dataset(self.opt, self.opt.dataset + '_' + 'noisy', type_of_data = type_of_data, args_for_dataset = args_for_dataset)
				self.encode_2_part_dataset(file_encoded_path,dataset_to_encode, self.opt.limit_conversion, y_classes = dataset_to_encode.dataset.num_classes,\
					compress_fly = False)

			#Normal dataset for evaluation
			type_of_data = 'test'
			file_encoded_path = os.path.join(self.opt.encoded_data_dir,'{}-encoded-{}.data'.format(self.opt.dataset, type_of_data))
			dataset_to_encode = create_dataset(self.opt, self.opt.dataset, type_of_data = type_of_data)
			self.encode_2_part_dataset(file_encoded_path, dataset_to_encode, self.opt.limit_conversion, y_classes = dataset_to_encode.dataset.num_classes,\
				compress_fly = False)
		else:
			raise Exception('unknown task_type: {}'.format(task_type))

	def create_impossible_test_set_for_land(self, task_type):
		#Create a dataset for the 'land' tasks, where fly is one and the one image given is 0 (impossible)
		if task_type not in ['bland', 'g7land', 'g4land']:
			return

		if not os.path.exists(self.opt.encoded_data_dir):
			os.mkdir(self.opt.encoded_data_dir)

		type_of_data = 'test'

		additional_constraint_on_data = lambda x_label,domain_x: True if domain_x == 'domain_b' and x_label == 0 else True if domain_x != 'domain_b' else False
			# not (domain_x == 'domain_b' and x_label == 1)
		relational_func = lambda a, b: True
		if task_type == 'bland':
			additional_constraint_on_data = lambda x_label,domain_x: not (domain_x == 'domain_b' and x_label == 1)
			domain_constraints = lambda label: label == 0 or label == 1
		else:
			if task_type   == 'g7land':
				additional_constraint_on_data = lambda x_label,domain_x: not (domain_x == 'domain_b' and x_label > 7)
			elif task_type == 'g4land':
				additional_constraint_on_data = lambda x_label,domain_x: not (domain_x == 'domain_b' and x_label > 4)
			domain_constraints = lambda label: True

		args_for_dataset = {'additional_constraint_on_data': additional_constraint_on_data, \
							'relational_func': relational_func,\
							'domain_constraints': domain_constraints}

		file_encoded_path = os.path.join(self.opt.encoded_data_dir,'{}_{}-encoded-{}_impossible.data'.format(self.opt.dataset, task_type, type_of_data))
		self.encode_logic_dataset(file_encoded_path,task_type,  type_of_data, self.opt.limit_conversion, args_for_dataset = args_for_dataset)

	def encode_specific_file(self, file_encoded_path, type_of_data = 'train', limit_conversion = -1, compress_fly = True):
		#Create specified dataset
		dataset_to_encode = create_dataset(self.opt, self.opt.dataset, type_of_data = type_of_data)

		# Set up network with the screenshot from the best performing batch
		self.load_net_at_best_epoch()

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

	def make_class_examples(self):
		pass


	def load_net_at_best_epoch(self):
		key = 'valid_{}'.format(self.opt.for_error.upper())
		if not key in self.best_val_model_idx:
			raise Exception('The network does not hold information for the provided error: {} (key: {})'.format(for_error, key))
		epoch_idx = self.best_val_model_idx[key]
		self.model.load_networks(epoch_idx)
		self.model.annealing_temp = self.annealing_temp_min

	def encode_successor_dataset(self, file_encoded_path, type_of_data = 'train', limit_conversion = -1):
		dataset_to_encode = create_dataset(self.opt, self.opt.dataset + '_succ', type_of_data = type_of_data)
		
		self.load_net_at_best_epoch()

		flx_compressed_var_length = int(np.ceil(np.log2(self.opt.categorical_dim)))
		flx_compressed_size = flx_compressed_var_length * self.opt.feature_layer_size

		fla_info = FlDomainInfo('fla', self.opt.feature_layer_size, self.opt.categorical_dim, True, 0, flx_compressed_size)
		flb_info = FlDomainInfo('flb', self.opt.feature_layer_size, self.opt.categorical_dim, True, flx_compressed_size, flx_compressed_size * 2)
		fl_info = {'fla':fla_info, 'flb':flb_info}

		stored_elements = 0
		total_wrt_batch = len(dataset_to_encode)

		if limit_conversion != -1:
			total_wrt_batch = min(total_wrt_batch, int(np.ceil(limit_conversion/self.opt.batch_size)))

		with tqdm.tqdm(total=total_wrt_batch) as pbar:
			for batch_idx, data in enumerate(dataset_to_encode):  # sample batch

				data_a = data['domain_a']
				data_b = data['domain_b']

				self.model.set_input(data_a)
				self.model.run_encoder()
				fla = self.model.feature_layer.detach().numpy() #.view(self.model.feature_layer.shape[0], -1)
				# fly_onehot = data['targets'].detach().numpy()

				self.model.set_input(data_b)
				self.model.run_encoder()
				flb = self.model.feature_layer.detach().numpy() #.view(self.model.feature_layer.shape[0], -1)
				# fly_onehot = data['targets'].detach().numpy()

				fl_encoded_size = write_fl_batch_to_file_new(file_encoded_path, {'fla':fla, 'flb':flb}, fl_info, batch_idx)
				pbar.update(1)

				stored_elements += self.opt.batch_size
				if limit_conversion != -1 and stored_elements >= limit_conversion:
					break

		print('[ENCODE]\t finished converting dataset: {} - size: ({},{}) \n\t\t to file: {}'.format(dataset_to_encode, stored_elements, fl_encoded_size, file_encoded_path))
		return True

	def encode_logic_dataset(self, file_encoded_path, task_type, type_of_data = 'train', limit_conversion = -1, y_classes = 2, args_for_dataset = {}):
		dataset_to_encode = create_dataset(self.opt, self.opt.dataset + '_logic', type_of_data = type_of_data, args_for_dataset = args_for_dataset)
		self.encode_3_part_dataset(file_encoded_path, dataset_to_encode, limit_conversion, y_classes)


	def encode_2_part_dataset(self, file_encoded_path, dataset_to_encode,limit_conversion, y_classes, compress_fly = True):
		self.load_net_at_best_epoch()

		flx_compressed_var_length = int(np.ceil(np.log2(self.opt.categorical_dim)))
		flx_compressed_size = flx_compressed_var_length * self.opt.feature_layer_size

		if compress_fly:
			fly_compressed_var_length = int(np.ceil(np.log2(dataset_to_encode.dataset.num_classes)))
		else:
			fly_compressed_var_length = dataset_to_encode.dataset.num_classes
		fly_compressed_size = fly_compressed_var_length * 1

		fla_info = FlDomainInfo('fla', self.opt.feature_layer_size, self.opt.categorical_dim, True, 0, flx_compressed_size)
		fly_info = FlDomainInfo('fly', 1, y_classes, compress_fly, flx_compressed_size, flx_compressed_size + fly_compressed_size)
		fl_info = {'fla':fla_info, 'fly':fly_info}

		stored_elements = 0
		total_wrt_batch = len(dataset_to_encode)

		if limit_conversion != -1:
			total_wrt_batch = min(total_wrt_batch, int(np.ceil(limit_conversion/self.opt.batch_size)))

		with tqdm.tqdm(total=total_wrt_batch) as pbar:
			for batch_idx, data in enumerate(dataset_to_encode):  # sample batch

				self.model.set_input(data)
				self.model.run_encoder()
				fla = self.model.feature_layer.detach().numpy() #.view(self.model.feature_layer.shape[0], -1)
				# fly_onehot = data['targets'].detach().numpy()
				fly = data['targets'].detach().numpy()

				fl_encoded_size = write_fl_batch_to_file_new(file_encoded_path, {'fla':fla, 'fly':fly}, fl_info, batch_idx)
				pbar.update(1)

				stored_elements += self.opt.batch_size
				if limit_conversion != -1 and stored_elements >= limit_conversion:
					break

		print('[ENCODE]\t finished converting dataset: {} - size: ({},{}) \n\t\t to file: {}'.format(dataset_to_encode, stored_elements, fl_encoded_size, file_encoded_path))
		return True


	def encode_3_part_dataset(self, file_encoded_path, dataset_to_encode, limit_conversion, y_classes):
		self.load_net_at_best_epoch()

		flx_compressed_var_length = int(np.ceil(np.log2(self.opt.categorical_dim)))
		flx_compressed_size = flx_compressed_var_length * self.opt.feature_layer_size

		if self.opt.compress_fly:
			fly_compressed_var_length = int(np.ceil(np.log2(y_classes)))
		else:
			fly_compressed_var_length = y_classes

		fla_info = FlDomainInfo('fla', self.opt.feature_layer_size, self.opt.categorical_dim, True, 0, flx_compressed_size)
		flb_info = FlDomainInfo('flb', self.opt.feature_layer_size, self.opt.categorical_dim, True, flx_compressed_size, flx_compressed_size * 2)
		fly_info = FlDomainInfo('fly', 1, y_classes, self.opt.compress_fly, flx_compressed_size * 2, flx_compressed_size * 2 + fly_compressed_var_length)
		fl_info = {'fla':fla_info,'flb': flb_info, 'fly':fly_info}

		stored_elements = 0
		total_wrt_batch = len(dataset_to_encode)

		if limit_conversion != -1:
			total_wrt_batch = min(total_wrt_batch, int(np.ceil(limit_conversion/self.opt.batch_size)))

		with tqdm.tqdm(total=total_wrt_batch) as pbar:
			for batch_idx, data in enumerate(dataset_to_encode):  # sample batch

				data_a = data['domain_a']
				data_b = data['domain_b']
				fly = data['y_label']

				self.model.set_input(data_a)
				self.model.run_encoder()
				fla = self.model.feature_layer.detach().numpy() #.view(self.model.feature_layer.shape[0], -1)
				# fly_onehot = data['targets'].detach().numpy()

				self.model.set_input(data_b)
				self.model.run_encoder()
				flb = self.model.feature_layer.detach().numpy() #.view(self.model.feature_layer.shape[0], -1)
				# fly_onehot = data['targets'].detach().numpy()

				fl_encoded_size = write_fl_batch_to_file_new(file_encoded_path, {'fla':fla, 'flb':flb, 'fly':fly}, fl_info, batch_idx)
				pbar.update(1)

				stored_elements += self.opt.batch_size
				if limit_conversion != -1 and stored_elements >= limit_conversion:
					break

		print('[ENCODE]\t finished converting dataset: {} - size: ({},{}) \n\t\t to file: {}'.format(dataset_to_encode, stored_elements, fl_encoded_size, file_encoded_path))
		return True
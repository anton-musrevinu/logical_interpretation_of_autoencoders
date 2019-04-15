import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
import shutil
from models import create_model
import time
import tqdm
import numpy as np

from util.storage_utils import save_to_stats_pkl_file, load_from_stats_pkl_file, \
	save_statistics, load_statistics, clean_model_dir
from util.plots import plot_stats_in_graph
from util.images import make_time_image

class BaseManager():

	def __init__(self, opt):

		"""

		Holds a network class over longer time, implementing the training, keeping track of the best
		epoch and is reponsible for loading and saving the model

		"""
		self.opt = opt
		self.network_id = self.get_manager_name()
		output_dir = os.path.abspath(self.opt.output_dir)

		#UNIQUE IDENTIFER FOR A GIVEN EXECUTION OF THE MAIN FAILE (DIR ALREADY EXISTS WITH OPT FILE)
		experiment_dir = os.path.abspath(self.opt.experiment_dir)
		# print(experiment_dir)


		self.manager_dir = os.path.join(experiment_dir,self.network_id)
		self.experiment_logs = os.path.join(self.manager_dir, "./result_outputs")
		self.experiment_saved_models = os.path.join(self.manager_dir, "./saved_models")
		self.experiment_training_log = os.path.abspath(os.path.join(self.experiment_logs, "summary.txt"))
		self.experiment_training_summary = os.path.abspath(os.path.join(self.manager_dir, "summary_training.txt"))
		self.opt.checkpoint_dir = self.manager_dir
		print('self.opt.checkpoint_dir ',self.opt.checkpoint_dir )

		self.create_model()

		self.best_val_model_idx = {}
		self.best_val_model_error = {}
		self.best_val_model_comarison = {}
		self.define_best_criterion()

		# if load_from_dir != 'None':
		# 	self.load_experiment(load_from_dir, experiment_name)
		if opt.phase == 'train':
			self.num_epochs = opt.num_epochs
			self.setup_training()
		else:
			self.load_training()

	def get_manager_name(self):
		return str(self.__class__.__name__)

	def define_best_criterion(self):
		if self.model.acc_names:
			for key in self.model.acc_names:
				key = 'valid_{}_acc'.format(key)
				self.best_val_model_idx[key] = 0
				self.best_val_model_error[key] = -float('Inf')
				self.best_val_model_comarison[key] = lambda current, best_so_far: current > best_so_far
		else:
			for key in self.model.loss_names:
				key = 'valid_{}'.format(key)
				self.best_val_model_idx[key] = 0
				self.best_val_model_error[key] = float('Inf')
				self.best_val_model_comarison[key] = lambda current, best_so_far: current < best_so_far

	def create_model(self):
		self.model = create_model(self.opt)
		self.model.setup(self.opt)  # re-initialize network parameters
		self.model.save_dir = self.experiment_saved_models


	### ----------------------------------------------------------------------------------------------------------
	### --------------------------------                UTILITIES        -----------------------------------------
	### ----------------------------------------------------------------------------------------------------------

	@staticmethod
	def modify_commandline_options(parser, is_train):
		"""Add new model-specific options, and rewrite default values for existing options.

		Parameters:
			parser          -- original option parser
			is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.
		"""

		parser.add_argument('--model_type', nargs="?", type=str, default=parser.model, help='Type of Autoencoder used: [base,lin, vae]')
		return parser

	def get_best_model_values_from_summery(self, upto = -1):
		num_of_computed_epochs = 0
		with open(self.experiment_training_log,'r') as f:
			self.total_losses = {}
			names = []
			for line_num,line in enumerate(f):
				row = line.strip().split(',')
				if line_num == 0:
					for idx,elem in enumerate(row):
						self.total_losses[elem] = []
						names.append(elem)
				elif upto == -1 or (line_num - 1) <= upto:
					for idx,elem in enumerate(row):
						self.total_losses[names[idx]].append(float(elem))
			# print(data)
			for key, value in self.total_losses.items():
				num_of_computed_epochs = len(value)
				if key in self.best_val_model_idx.keys():
					self.best_val_model_idx[key] = np.argmin(value) + 1
					self.best_val_model_error[key] = np.min(value)

		print('best_val_model_idx: {}, best_val_model_error: {}'.format(self.best_val_model_idx, self.best_val_model_error))
		print('num_of_computed_epochs', num_of_computed_epochs)
		return num_of_computed_epochs

	def load_training(self):

		if not os.path.exists(self.manager_dir):
			raise Exception('trying to load experiment, but {} does not exist'.format(self.manager_dir))
		if not os.path.exists(self.experiment_logs):
			raise Exception('trying to load experiment, but {} does not exist'.format(self.experiment_logs))
		if not os.path.exists(self.experiment_saved_models):
			raise Exception('trying to load experiment, but {} does not exist'.format(self.experiment_saved_models))

		self.get_best_model_values_from_summery()


	def setup_training(self):

		if self.opt.epoch_count == 1:
			# raise Exception
			os.mkdir(self.manager_dir)  # create the experiment directory
			os.mkdir(self.experiment_logs)  # create the experiment log directory
			os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory
			self.total_losses = {}

			for data_type in ['train', 'valid']:
				for kind in self.model.loss_names:
					name = '{}_{}'.format(data_type, kind)
					self.total_losses[name] = []
				if data_type == 'valid':
					for kind in self.model.acc_names:
						name = '{}_{}_acc'.format(data_type, kind)
						self.total_losses[name] = []

		elif self.opt.epoch_count == -1:
			self.opt.epoch_count = self.get_best_model_values_from_summery() + 1
			self.model.load_networks('latest')
			print('loading lates network (epoch_idx: {}) and continuing from epoch {}'.format(self.opt.epoch_count -1, self.opt.epoch_count))
			self.experiment_training_summary = self.experiment_training_summary.replace('summary_training', 'summary_training_cont_{}'.format(self.opt.epoch_count))
		# elif self.opt.epoch_count < 1:
		# 	for key in self.best_val_model_idx.keys():
		# 		if 'acc' in key:
		# 			key_value = 'best_{}'.format(key)
		# 			print('loading network {}'.format(key_value))
		# 			self.model.load_networks(key_value)
		# 			self.network_loaded = True
		# 	#IF NO BEST HAS BEEN COMPUTED
		# 	if not self.network_loaded:
		# 		self.get_best_model_values_from_summery()
		# 		for loss_type, best_val_model_idx_for_loss in self.best_val_model_idx.items():
		# 			print('loading network {} based on loss type: {}'.format(best_val_model_idx_for_loss, loss_type))
		# 			self.model.load_networks(best_val_model_idx_for_loss)
		# 			self.network_loaded = True
		# 	self.opt.epoch_count = 1
		else:
			self.get_best_model_values_from_summery(upto = self.opt.epoch_count)
			self.model.load_networks(self.opt.epoch_count)
			self.experiment_training_summary = self.experiment_training_summary.replace('training_summery', 'summary_training_cont_{}'.format(self.opt.epoch_count))


	### ----------------------------------------------------------------------------------------------------------
	### --------------------------------               TRAINING        -----------------------------------------
	### ----------------------------------------------------------------------------------------------------------


	def run_train_iter(self, data):
		"""
		Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
		:param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
		:param y: The targets for the model. A numpy array of shape batch_size, num_classes
		:return: the loss and accuracy for this batch
		"""
		self.model.train()
		self.model.set_input(data)         # unpack data from dataset and apply preprocessing
		self.model.optimize_parameters()

		return self.model.get_current_losses()

	def after_training_updates(self,epoch_idx, batch_idx, losses):

		return True

	def record_additional_info(self, epoch_idx):

		return True

	def run_evaluation_iter(self, data):
		"""
		Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
		:param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
		:param y: The targets for the model. A numpy array of shape batch_size, num_classes
		:return: the loss and accuracy for this batch
		"""
		self.model.eval()
		self.model.set_input(data)
		self.model.compute_network()

		t = self.model.get_current_losses()
		# print(t)
		t.update(self.model.get_current_accs())
		# print(t)

		return t

	def run_testing_iter(self, data, results,total_pos_examples,total_neg_examples):
		"""
		Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
		:param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
		:param y: The targets for the model. A numpy array of shape batch_size, num_classes
		:return: the loss and accuracy for this batch
		"""
		self.model.set_input(data)
		results_comp,predicted_idxs, real_idxs = self.model.compute_network()

		for i in range(len(data['indexs'])):
			if not results_comp[i] and total_neg_examples > 0:
				results[data['indexs'][i]] = (results_comp[i],predicted_idxs[i], real_idxs[i])
				total_neg_examples -= 1
			elif results_comp[i] and total_pos_examples > 0:
				results[data['indexs'][i]] = (results_comp[i],predicted_idxs[i], real_idxs[i])
				total_pos_examples -= 1
			elif total_pos_examples <= 0 and total_neg_examples <= 0:
				break

		t = self.model.get_current_losses()
		# print(t)
		t.update(self.model.get_current_accs())
		# print(t)

		return t,results,total_pos_examples,total_neg_examples

	def get_losses_as_string(self,losses):
		loss_as_str = ''
		for loss_term in losses.values():
			loss_as_str = loss_as_str + '{:.4f},'.format(loss_term)
		return loss_as_str[:-1]

	def do_training(self):

		# self.total_losses = {}
		total_iters = 0
		start_time = time.time()
		
				# initialize a dict to keep the per-epoch metrics

		print(self.total_losses)
		for i, epoch_idx in enumerate(range(self.opt.epoch_count, self.num_epochs + 1)):
			epoch_start_time = time.time()
			current_epoch_losses = {}
			for key in self.total_losses.keys():
				current_epoch_losses[key] = []

			#--- Do training (for evaluation data)
			with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
				for batch_idx, data_batch in enumerate(self.train_data):  # get data batches

					#Computing the loss, backprobakating and updating the weights
					losses = self.run_train_iter(data_batch)  # take a training iter step

					#Manager Specific updates such as anneling
					self.after_training_updates(epoch_idx,batch_idx,losses)

					#Update current epoch losses
					for kk, vv in losses.items():
						kk_2 = 'train_{}'.format(kk)
						current_epoch_losses[kk_2].append(vv)

					#Update Progress Bar
					loss_as_str = self.get_losses_as_string(losses)
					pbar_train.set_description(' ' * 8 + "loss: {}".format(loss_as_str))
					total_iters += 1
					pbar_train.update(1)

			#--- Do evaluation (for evaluation data)
			with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
				for data_batch in self.val_data:  # get data batches

					#Computing the loss on validation batch
					losses_acc = self.run_evaluation_iter(data_batch)


					#Computing the loss, backprobakating and updating the weights
					for kk, vv in losses_acc.items():
						kk_2 = 'valid_{}'.format(kk)
						current_epoch_losses[kk_2].append(vv)

					#Update Progress Bar
					loss_as_str = self.get_losses_as_string(losses_acc)
					pbar_val.set_description(' ' * 8 + "loss: {}".format(loss_as_str))
					pbar_val.update(1)

		#End of training epoch
		#Recording losses, data, images and statistics

			#Record best the best model on the validation set for each acc type
			for key, comparefunc in self.best_val_model_comarison.items():
				val_mean_error = np.mean(current_epoch_losses[key])

				if comparefunc(val_mean_error, self.best_val_model_error[key]):# if current epoch's mean val acc is greater than the saved best val acc then
					self.best_val_model_error[key] = val_mean_error  # set the best val model acc to be current epoch's val accuracy
					self.best_val_model_idx[key] = epoch_idx

			#Update total losses
			for key, value in current_epoch_losses.items():
				self.total_losses[key].append(np.mean(value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

			#Save statistics
			save_statistics(summary_filename=self.experiment_training_log,
							stats_dict=self.total_losses, current_epoch=epoch_idx)  # save statistics to stats file.

			# load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

			#Display current statistics
			out_string = "_".join(
				["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
			# create a string to use to report our epoch metrics
			epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
			epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
			print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")

			# cache our model every <save_epoch_freq> epochs
			if epoch_idx % self.opt.save_epoch_freq == 0 and not self.opt.testing:              
				print('\t - saving the model at the end of epoch %d, iters %d' % (epoch_idx, total_iters))
				self.model.save_networks('latest')
				self.model.save_networks(epoch_idx)

			iter_data_time = time.time()

			self.record_additional_info(epoch_idx)

		train_time = time.time() - start_time

		# for loss_type, best_val_model_idx_for_loss in self.best_val_model_idx.items():
		# 	loss_type_key = 'test_{}_acc'.format(loss_type)

		# 	print("Saving best model based on: {}".format(loss_type_key))
		# 	self.model.load_networks(best_val_model_idx_for_loss)
		# 	self.model.save_networks('best_{}'.format(loss_type))

		if not self.best_val_model_idx:
			clean_model_dir(self.experiment_saved_models,[epoch_idx])
		else:
			clean_model_dir(self.experiment_saved_models,self.best_val_model_idx.values())

		with open(self.experiment_training_summary,'w') as f:
			f.write('training finished successfully\n')
			f.write('training time: {}\n'.format(train_time))
			f.write('training epochs: {}\n'.format(self.num_epochs))
			f.write('training data set: {}\n'.format(self.train_data))
			f.write('valididation data set: {}\n'.format(self.val_data))
			for key, idx in self.best_val_model_idx.items():
				f.write('for error: {}\n'.format(key))
				f.write('\tbest model: {}\n'.format(idx))
				# f.write('\tcorresponding value: {}\n'.format(self.best_val_model_error[key]))
				out_string = " ".join(
					["\t\t{}:{:.4f}\n".format(key2, value[idx - 1]) for key2, value in self.total_losses.items()])
				f.write('\tall corresponding errors: \n{}'.format(out_string))

		plot_stats_in_graph(self.total_losses, self.manager_dir)
		make_time_image(self.experiment_logs)

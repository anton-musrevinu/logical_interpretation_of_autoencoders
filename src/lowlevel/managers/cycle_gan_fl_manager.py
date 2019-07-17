
from .msc_manager import MSCManager

from models import create_model
import torch
import numpy as np
import os
from util.storage_utils import save_example_image, save_feature_layer_example

class CycleGanFLManager(MSCManager):

	def __init__(self,opt, train_data, val_data):

		MSCManager.__init__(self,opt)


	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		"""Add new dataset-specific options, and rewrite default values for existing options.

		Parameters:
			parser          -- original option parser
			is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.
		"""

		parser = MSCManager.modify_commandline_options(parser, is_train = is_train)

		return parser

	def _save_example_batch(self,epoch_idx):
		save_stuff = [self.model.real_A, self.model.fake_B, self.model.rec_A,self.model.real_B,self.model.fake_A,self.model.rec_B]
		row = map(lambda img_batch: img_batch.cpu().float(), save_stuff)
		row_np = torch.cat(list(row), 1)
		path = os.path.join(self.experiment_logs, 'transfer_example_epoch_{}.png'.format(epoch_idx))
		print(row_np.shape)
		save_example_image(row_np,path, nrow = row_np.shape[1])
		self.saved_epochs.append(epoch_idx)

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


	# def convert_target_to_source(self,dataSet, source_identifier):
		transfer_result = {}
		for key, best_val_model_idx_for_loss in self.best_val_model_idx.items():
			#Load the best model for the given key
			self.model.load_networks(best_val_model_idx_for_loss)




			batches_inputs = []
			batches_targets = []
			batches_input_idx = []
			data_id = '{}-transfered-{}'.format(dataSet.dataset.type_of_data, key)
			newdir = os.path.join(self.experiment_logs,'{}-{}.npz'.format(source_identifier,data_id))

			for data in dataSet:  # sample batch
				inputs_batch = data['inputs']
				target_batch = data['targets']

				fake, rec = self.model.transfer(inputs_batch, to_domain = source_identifier)
				fake = fake.cpu().detach().numpy()
				batches_inputs.append(fake)
				batches_targets.append(np.argmax(target_batch, axis = 1))
				batches_input_idx.append(data['indexs'])

			batches_all_inputs = np.concatenate(batches_inputs, axis = 0)
			batches_all_inputs = np.reshape(batches_all_inputs,(batches_all_inputs.shape[0], -1))
			batches_all_targets = np.concatenate(batches_targets, axis = 0)
			batches_input_idx = np.concatenate(batches_input_idx, axis = 0)
			# print(batches_all_inputs.shape, batches_all_targets.shape)

			np.savez(newdir, inputs=batches_all_inputs, targets=batches_all_targets)
			print('* translated the dataSet {} to {} and saved it at {}'.format(dataSet, data_id, self.experiment_logs))
			transfer_result[key] = batches_input_idx

		return self.experiment_logs, transfer_result


from .base_manager import BaseManager

from models import create_model
import torch
import numpy as np
import os
from util.storage_utils import save_example_image

class CycleGanManager(BaseManager):

	def __init__(self,opt, train_data, val_data):


		self.train_data = train_data
		self.val_data = val_data

		opt.input_nc = train_data.dataset.image_num_channels
		opt.output_nc = opt.input_nc

		BaseManager.__init__(self,opt)

		self.saved_epochs = []

	def _save_example_batch(self,epoch_idx):
		save_stuff = [self.model.real_A,self.model.fake_B,self.model.rec_A,self.model.real_B,self.model.fake_A,self.model.rec_B]
		row = map(lambda img_batch: img_batch.cpu().float(), save_stuff)
		row_np = torch.cat(list(row), 1)
		path = os.path.join(self.experiment_logs, 'transfer_example_epoch_{}.png'.format(epoch_idx))
		print(row_np.shape)
		save_example_image(row_np,path, nrow = row_np.shape[1])
		self.saved_epochs.append(epoch_idx)

	def run_evaluation_iter(self, data, epoch_idx):
		"""
		Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
		:param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
		:param y: The targets for the model. A numpy array of shape batch_size, num_classes
		:return: the loss and accuracy for this batch
		"""
		self.model.eval()
		self.model.set_input(data)
		self.model.compute_network()

		if epoch_idx % self.opt.save_epoch_freq == 0 and not epoch_idx in self.saved_epochs:
			self._save_example_batch(epoch_idx)

		t = self.model.get_current_losses()
		# print(t)
		t.update(self.model.get_current_accs())
		# print(t)

		return t

	def run_train_iter(self, data,total_iters):
		"""
		Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
		:param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
		:param y: The targets for the model. A numpy array of shape batch_size, num_classes
		:return: the loss and accuracy for this batch
		"""
		self.model.train()
		self.model.set_input(data)         # unpack data from dataset and apply preprocessing
		self.model.optimize_parameters(total_iters % self.opt.critic_iter == 0)

		return self.model.get_current_losses()

	def convert_target_to_source(self,dataSet, source_identifier):
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

	def create_model(self):
		# print(self.opt.input_nc, self.opt.output_nc)
		self.opt.model = 'cycle_gan'
		self.model = create_model(self.opt)
		self.model.setup(self.opt)  # re-initialize network parameters
		self.model.save_dir = self.experiment_saved_models

	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		"""Add new dataset-specific options, and rewrite default values for existing options.

		Parameters:
			parser          -- original option parser
			is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.

		For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
		A (source domain), B (target domain).
		Generators: G_A: A -> B; G_B: B -> A.
		Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
		Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
		Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
		Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
		Dropout is not used in the original CycleGAN paper.
		"""

		# parser.add_argument('--classifier_name', nargs="?", type=str, default="classifier",
		#                     help='Experiment name - to be used for building the experiment folder')
		# parser.add_argument('--replace_existing', nargs="?",type=self.str2bool, default=False,
		#                     help='Specify if an exerpiment directory with the same name should be overwritten or not')
		# parser.add_argument('--load', nargs="?", type=str, default="None",
		#                         help='Experiment folder to loads')

		return parser

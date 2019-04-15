
from .base_manager import BaseManager

from models import create_model
import time
import tqdm
import numpy as np
import os

class ClassifierManager(BaseManager):

	def __init__(self,opt, train_data, val_data, domain):

		self.domain_trained_for = domain
		self.train_data = train_data
		self.val_data = val_data

		self.input_shape = (opt.batch_size, self.train_data.dataset.image_num_channels,
							self.train_data.dataset.image_width,self.train_data.dataset.image_height)

		BaseManager.__init__(self,opt)

	def get_manager_name(self):
		return str(self.__class__.__name__) + '-{}'.format(self.domain_trained_for)


	def create_model(self):
		# print('creating model with input shape :', self.input_shape)
		self.opt.model = 'classifier'
		self.model = create_model(self.opt, self.input_shape)
		self.model.setup(self.opt)  # re-initialize network parameters
		self.model.save_dir = self.experiment_saved_models	

	def print_save_iterational_updates(self,epoch_idx,total_iters,iter_start_time):
		if total_iters % self.opt.print_freq == 0:    # print training losses and save logging information to the disk
			losses = self.model.get_current_losses()
			t_comp = (time.time() - iter_start_time) / self.opt.batch_size
			print('epoch: {}, total_iter: {}, losses: {}, t_comp: {}'.format(\
				epoch_idx,total_iters, losses, t_comp))

		if total_iters % self.opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
			print('saving the latest model (epoch %d, total_iters %d)' % (epoch_idx, total_iters))
			save_suffix = 'iter_%d' % total_iters if self.opt.save_by_iter else 'latest'
			self.model.save_networks(save_suffix)

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

	def do_evaluation(self, dataSet, datatype = ''):

		overall_test_losses_best_for = {}
		total_pos_examples = 25
		total_neg_examples = 25

		start_time = time.time()

		print("Generating the given model on the dataset: {}".format(dataSet))

		for loss_type, best_val_model_idx_for_loss in self.best_val_model_idx.items():

			self.model.load_networks(best_val_model_idx_for_loss)

			current_epoch_losses = {}
			for key in self.model.loss_names:
				key_str = 'test_' + key
				current_epoch_losses[key_str] = []
			for key in self.model.acc_names:
				key_str = 'test_' + key + '_acc'
				current_epoch_losses[key_str] = []

			results = {}
			# current_epoch_losses = {"test_loss": [],'test_recon_loss': [],'test_kld_loss': []}  # initialize a statistics dict
			with tqdm.tqdm(total=len(dataSet)) as pbar_test:  # ini a progress bar
				for data in dataSet:  # sample batch
					losses_acc,results,total_pos_examples,total_neg_examples = \
							self.run_testing_iter(data, results,total_pos_examples,total_neg_examples)  # run a validation iter

					for kk, vv in losses_acc.items():
						kk_2 = 'test_{}'.format(kk)
						current_epoch_losses[kk_2].append(vv)

					pbar_test.update(1)

					loss_as_str = ''
					for loss_term in losses_acc.values():
						loss_as_str = loss_as_str + '{:.4f},'.format(loss_term)
					pbar_test.set_description(' ' * 8 + "loss: {}".format(loss_as_str[:-1]))

			test_losses = {key: [np.mean(value)] for key, value in
						   current_epoch_losses.items()}  # save test set metrics in dict format
			print('\tbest model for loss: {} -> achived error: {}'.format(loss_type, test_losses))
			overall_test_losses_best_for[loss_type] = test_losses

		total_time = time.time() - start_time

		#results: dictionary.. key: index of data point wrt to dataSet, value true/false classification

		new_test_file = self.experiment_training_summary.replace('training', 'testing')
		
		if os.path.exists(new_test_file):
			write_append = 'a'
		else:
			write_append = 'w'

		with open(new_test_file,write_append) as f:
			if write_append == 'w':
				f.write('------- Testing results for Classifier, trained on domain: {} -------\n\n'.format(\
					self.domain_trained_for))
			f.write('testing dataset: {}\n'.format(dataSet))
			f.write('testing domain: {}\n'.format(datatype))
			f.write('testing time: {}\n'.format(total_time))
			for key, value in overall_test_losses_best_for.items():
				f.write('for error: {}\n'.format(key))
				f.write('\ttesting results are: {}\n'.format(value))
			f.write('=' * 20 + '\n\n')

		return results
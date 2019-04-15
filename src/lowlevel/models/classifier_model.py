import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import os


class ClassifierModel(BaseModel):

	@staticmethod
	def str2bool(self,v):
		if v.lower() in ('yes', 'true', 't', 'y', '1'):
			return True
		elif v.lower() in ('no', 'false', 'f', 'n', '0'):
			return False
		else:
			raise argparse.ArgumentTypeError('Boolean value expected.')


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
		parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
		# if is_train:

		parser.add_argument('--dim_reduction_type', nargs="?", type=str, default='strided_convolution',
							help='One of [strided_convolution, dilated_convolution, max_pooling, avg_pooling, alex_net]')
		parser.add_argument('--num_layers', nargs="?", type=int, default=4,
							help='Number of convolutional layers in the network (excluding '
								 'dimensionality reduction layers)')
		parser.add_argument('--num_output_classes', nargs="?", type=int, default=10,
							help='Number of classes to classify')
		parser.add_argument('--use_bias',nargs="?", type=ClassifierModel.str2bool, default=True, help='use_bias a bias should be used')
		parser.add_argument('--num_filters', nargs="?", type=int, default=64,
							help='Number of convolutional filters per convolutional layer in the network (excluding '
								 'dimensionality reduction layers)')
		parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=1e-05,help='Weight decay to use for Adam')
		# parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

		return parser


	def __init__(self, opt, input_shape = None):
		"""Initialize the CycleGAN class.

		Parameters:
			opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		BaseModel.__init__(self, opt)
		# specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
		self.loss_names = ['C']
		self.acc_names = ['C']
		# specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
		visual_names_A = []#['real_A', 'fake_B', 'rec_A']
		visual_names_B = []#['real_B', 'fake_A', 'rec_B']
		# if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
		#     visual_names_A.append('idt_B')
		#     visual_names_B.append('idt_A')

		self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
		# specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
		if self.isTrain:
			self.model_names = ['SC']
		else:  # during test time, only load Gs
			self.model_names = ['SC']

		# define networks (classifier for images)

		self.netSC = networks.define_C(input_shape, opt.num_filters, opt.num_layers, opt.dim_reduction_type,
										opt.num_output_classes,
										opt.use_bias,
										norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[])

		self.netSC.to(self.device)

		if self.isTrain:
			# define loss functions
			self.criterionC = torch.nn.CrossEntropyLoss().to(self.device)
			# initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
			self.optimizer_C = torch.optim.Adam(self.netSC.parameters(), amsgrad=False, weight_decay=opt.weight_decay_coefficient)#, lr=opt.lr)
			# self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizers.append(self.optimizer_C)


	def save_networks(self, epoch):
		"""Save all the networks to the disk.

		Parameters:
			epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
		"""
		for name in self.model_names:
			if isinstance(name, str):
				save_filename = '%s_net_%s.pth' % (epoch, name)
				save_path = os.path.join(self.save_dir, save_filename)
				net = getattr(self, 'net' + name)

				if len(self.gpu_ids) > 0 and torch.cuda.is_available():
					torch.save(net.cpu().state_dict(), save_path)
					net.cuda(self.gpu_ids[0])
				else:
					torch.save(net.cpu().state_dict(), save_path)


	def set_input(self, input):
		"""Unpack input data from the dataloader and perform necessary pre-processing steps.
			Basically the batch to be compted

		Parameters:
			input (dict): include the data itself and its metadata information.

		The option 'direction' can be used to swap domain A and domain B.
		"""
		self.input = input['inputs'].to(self.device)
		self.target = input['targets'].to(self.device)
		# AtoB = self.opt.direction == 'AtoB'
		# self.real_A = input['A' if AtoB else 'B'].to(self.device)
		# self.real_B = input['B' if AtoB else 'A'].to(self.device)
		# self.image_paths = input['A_paths' if AtoB else 'B_paths']

	def forward(self):
		"""Run forward pass; called by both functions <optimize_parameters> and <test>."""
		self.predicted = self.netSC(self.input)  # G_A(A)

	def optimize_parameters(self):
		"""Calculate losses, gradients, and update network weights; called in every training iteration"""
		# forward
		self.forward()      # compute fake images and reconstruction images.

		_, test = torch.max(self.target,1)
		# test = torch.Tensor(test).float().to(self.device)
		self.loss_C = self.criterionC(self.predicted.float(), test)  # compute loss

		self.optimizer_C.zero_grad()  # set all weight grads from previous training iters to 0
		self.loss_C.backward()  # backpropagate to compute gradients for current iter loss

		self.optimizer_C.step()  # update network parameters

		# _, predicted = torch.max(self.predicted.data, 1)  # get argmax of predictions
		# accuracy = np.mean(list(predicted.eq(test.data).cpu()))  # compute accuracy
		# self.loss_C_ACC = accuracy

	def compute_network(self):
		self.forward()

		_, test = torch.max(self.target,1)
		self.loss_C = self.criterionC(self.predicted.float(), test)  # compute loss

		_, predicted = torch.max(self.predicted.data, 1)  # get argmax of predictions
		accuracy = np.mean(list(predicted.eq(test.data).cpu()))  # compute accuracy
		self.acc_C = accuracy

		return predicted.eq(test.data).cpu(), predicted.cpu(), test.data.cpu()

		# return loss.data.cpu().numpy(), accuracy

  #       # G_A and G_B
  #       self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
  #       self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
  #       self.backward_G()             # calculate gradients for G_A and G_B
  #       self.optimizer_G.step()       # update G_A and G_B's weights
  #       # D_A and D_B
  #       self.set_requires_grad([self.netD_A, self.netD_B], True)
  #       self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
  #       self.backward_D_A()      # calculate gradients for D_A
  #       self.backward_D_B()      # calculate graidents for D_B
  #       self.optimizer_D.step()  # update D_A and D_B's weights

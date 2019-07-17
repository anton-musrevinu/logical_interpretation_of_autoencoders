from .networks import *
from .autoencoders.var_autoencoders import VarLinAutoEncoder,VarConvAutoEncoder,VarResNetAutoEncoder
import torch

def define_AE(opt):

	net = None
	net = VarConvAutoEncoder(opt)

	if len(opt.gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(opt.gpu_ids[0])

	if not opt.loading:
		net = init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
	return net

def define_encoder(opt):
	net = VarConvEncoder(opt)
	shape_args = (net.conversion_layer_shape_before, net.conversion_layer_shape_after)

	if len(opt.gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(opt.gpu_ids[0])

	if not opt.loading:
		net = init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
		
	return net, shape_args

def define_decoder(opt, shape_args):
	net = VarConvDecoder(opt,shape_args = shape_args)

	if len(opt.gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(opt.gpu_ids[0])

	if not opt.loading:
		net = init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
	return net

def define_resNetAE(opt):

	net = None
	net = VarResNetAutoEncoder(opt)

	if len(opt.gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(opt.gpu_ids[0])

	if not opt.loading:
		net = init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
	return net

def define_flDisciminator(opt):
	norm_layer = get_norm_layer(norm_type=opt.norm)

	net = NLayerFLDiscriminator(opt.feature_layer_size, opt.categorical_dim , n_layers=6, norm_layer=norm_layer)

	if len(opt.gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(opt.gpu_ids[0])

	if not opt.loading:
		net = init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
	return net

#=============================================================================================================================================
#=============================================================================================================================================
#=============================================================================================================================================
class NLayerFLDiscriminator(nn.Module):
	"""Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

	def __init__(self, feature_layer_size, categorical_dim, n_layers, norm_layer=nn.BatchNorm2d):
		"""Construct a 1x1 PatchGAN discriminator

		Parameters:
			input_nc (int)  -- the number of channels in input images
			ndf (int)       -- the number of filters in the last conv layer
			norm_layer      -- normalization layer
		"""
		super(NLayerFLDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
			use_bias = norm_layer.func != nn.InstanceNorm2d
		else:
			use_bias = norm_layer != nn.InstanceNorm2d

		self.net = []
		current_dim = feature_layer_size * categorical_dim
		layer_sizes = list(map(int,np.linspace(1, current_dim, n_layers)))[::-1]
		for i in range(n_layers):
			self.net.append(nn.Linear(layer_sizes[i], layer_sizes[i+1],bias=True))
			if i < n_layers - 1:
				self.net.append(nn.LeakyReLU(0.2, True))
		self.net.append(nn.Sigmoid())

		self.net = nn.Sequential(*self.net)

	def forward(self, input):
		"""Standard forward."""
		return self.net(input)


#=============================================================================================================================================
#=============================================================================================================================================


class Gumbell_kld(torch.nn.Module):

	def __init__(self, beta, categorical_dim, fl_flat_shape):

		super(Gumbell_kld, self).__init__()
		self.beta = beta
		self.eps = torch.tensor(1e-20)
		self.fl_flat_shape = fl_flat_shape
		self.categorical_dim = categorical_dim

	def __call__(self, feature_layer_prob):
		# print('feature_layer_prob.shape',feature_layer_prob.shape)
		categorical_shape = feature_layer_prob.shape
		softmax = torch.nn.Softmax(dim=-1)

		qy = softmax(feature_layer_prob).view(self.fl_flat_shape)
		# print('qy.shape',qy.shape)

		# kl1 = qy * torch.log(qy + self.eps)
		# kl2 = qy * torch.log(1.0/self.categorical_dim  + self.eps)

		# kld_loss_term = torch.sum(torch.sum(kl1 - kl2, 2), 1)

		# q_y = logistic_fl.view(self.fl_flat_shape)
		# qy = F.softmax(q_y, dim=-1).view(self.fl_flat_shape)
		# # print(logistic_fl.shape, qy.shape)
		log_ratio = torch.log(qy * self.categorical_dim + self.eps)
		kld_loss_term = torch.sum((qy * log_ratio).view(categorical_shape), dim=-1).mean()

		kld_loss = self.beta * kld_loss_term
		# print(kld_loss)
		return kld_loss

def sample_gumbel(shape, device, eps=1e-20):
	U = torch.rand(shape).to(device)
	return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, device):
	y = logits + sample_gumbel(logits.size(),device)
	softmax = torch.nn.Softmax(dim=-1)
	return softmax(y/temperature)

def gumbel_softmax(logits , device, temperature = 1, hard=False):
	"""
	ST-gumple-softmax
	input: [batch_size, num_variables, categorical_dim]
	return: flatten --> [batch_size, n_class] an one-hot vector
	"""
	fl_hidden_shape = logits.shape
		#shape[0] = batchsize
		#shape[1] = fl_size
		#shape[2] = categorical dim


	# q_y = logits.view(categorical_shape)
	# print('logits.shape',logits.shape)
	y = gumbel_softmax_sample(logits, temperature, device)

	if not hard:
		return y

	shape = y.size()
	_, ind = y.max(dim=-1)
	y_hard = torch.zeros_like(y).view(-1, shape[-1])
	y_hard.scatter_(1, ind.view(-1, 1), 1)
	y_hard = y_hard.view(*shape)
	# Set gradients w.r.t. y_hard gradients w.r.t. y
	y_hard = (y_hard - y).detach() + y
	return y_hard
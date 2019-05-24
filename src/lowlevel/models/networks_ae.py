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


def define_resNetAE(opt):

	net = None
	net = VarResNetAutoEncoder(opt)

	if len(opt.gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(opt.gpu_ids[0])

	if not opt.loading:
		net = init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
	return net
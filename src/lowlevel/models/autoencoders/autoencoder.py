import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import functools
# from ..distributions.gumbel import gumbel_softmax

class Autoencoder(nn.Module):
    def __init__(self, input_shape,feature_layer_size, use_bias=False):
        super(Autoencoder, self).__init__()
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param dim_reduction_type: The type of dimensionality reduction to apply after each convolutional stage, should be one of ['max_pooling', 'avg_pooling', 'strided_convolution', 'dilated_convolution']
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        :param num_layers: Number of conv layers (excluding dim reduction stages)
        :param use_bias: Whether our convolutions will use a bias.
        """
        # set up class attributes useful 10in building the network and inference
        self.input_shape = input_shape
        self.feature_layer_size = feature_layer_size
        self.use_bias = use_bias
        self.layer_dict = nn.ModuleDict()
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        # build the network

        self.build_module()

    def build_module(self):
        raise Exception('Not Implemented Funcion')

    def forward(self, x):
        out = x
        out = out.view(out.shape[0], -1)

        encoder_out = self.encoder(out)

        x = self.decoder(encoder_out)
        res = x.view(self.input_shape)

        feature_layer = encoder_out.data.numpy()

        return res, feature_layer, regul_loss

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.encoder.children():
            try:
                item.reset_parameters()
            except:
                pass
        for item in self.decoder.children():
            try:
                item.reset_parameters()
            except:
                pass

class ConvAutoencoder(Autoencoder):

    def __init__(self, input_shape,feature_layer_size,feature_leyer_act_scalar = 1,
                        use_bias=False, final_activation_str = 'tanh',num_layers = 2,
                        num_channels = 64, device = None, dropout = False):

        self.feature_leyer_act_scalar = feature_leyer_act_scalar
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.input_shapes = {}
        self.conversion_layer = 1
        self.num_fc = 1
        self.device = device
        self.final_activation_str = final_activation_str
        self.dropout = dropout

        super(ConvAutoencoder, self).__init__(input_shape,feature_layer_size,use_bias)

    def __str__(self):
        old = super(ConvAutoencoder, self).__str__()

        self.num_params = 0
        params_str = ''
        for i,param in enumerate(list(self.parameters())):
            params_str = params_str + \
                        '\n\t param: {} - shape: {} - requires_grad: {}'.format(i,param.shape, param.requires_grad)
            self.num_params += param.numel()

        name = 'MODEL CLASS: {}'.format(self.__class__.__name__)
        add = '\n input shape: {}'.format(self.input_shape) + \
              '\n feature layer size: {}'.format(self.feature_layer_size) + \
              '\n feature layer activation scalar: {}'.format(self.feature_leyer_act_scalar) + \
              '\n feature num_channels: {}'.format(self.num_channels) + \
              '\n feature num_layers: {}'.format(self.num_layers) + \
              '\n use_bias {}'.format(self.use_bias) + \
              '\n final_activation_str: {}'.format(self.final_activation_str) + \
              '\n model params: nbtensors - {}, nbparams - {}'.format(len(list(self.parameters())), self.num_params)
        add = add + params_str
        return old + name + add

    def build_module(self):

        raise Exception('Not implemented Function: build_module')

    def encode(self,x):
        out = x

        for i in range(self.num_layers_encoder):
            # print('{} - A : {}'.format(i,out.shape))
            if i == self.num_layers_encoder - self.num_fc:
                # if out.shape[-1] != 2:
                #     out = F.adaptive_avg_pool2d(out, 2)  # apply adaptive pooling to make sure output of conv layers is always (2, 2) spacially (helps with comparisons).
                out = out.view(self.conversion_layer_shape_after)
            # print('{} - B : {}'.format(i,out.shape))
            out = self.layer_dict['encode_{}'.format(i)](out)
            out = self.layer_dict['encode_{}_activation'.format(i)](out)
            # print('{} - C : {}'.format(i,out.shape))

            if 'encode_{}_reduction'.format(i) in self.layer_dict:
                out = self.layer_dict['encode_{}_reduction'.format(i)](out)

        return out

    def loss_function(self,recon_x,x, pre_encoder_out):
        MSE = nn.MSELoss().to(self.device)(recon_x, x)

        return MSE

    def decode(self,x):
        out = x

        for i in range(self.num_layers_decoder):
            # print('{} - A : {}'.format(i,out.shape))
            if i == self.num_fc:
                out = out.view(self.conversion_layer_shape_before)
            # print('{} - B : {}'.format(i,out.shape))
            out = self.layer_dict['decode_{}'.format(i)](out)

            out = self.layer_dict['decode_{}_activation'.format(i)](out)

            if 'decode_{}_upsampling'.format(i) in self.layer_dict:
                out = self.layer_dict['decode_{}_upsampling'.format(i)](out)

        res = out.view(self.input_shape)
        return res

    def final_activation(self,encoder_pre_out, activation = None):

        if activation == None:
            activation = self.final_activation_str

        if activation == 'tanh':
            feature_layer = encoder_pre_out
            regul_loss = 0
        elif activation ==  'heaviside':
            encoder_pre_out_np = encoder_pre_out.data.cpu().numpy()
            feature_layer_np = np.heaviside(encoder_pre_out_np, 1) * 2 -1
            regul_loss = np.sum(((feature_layer_np - encoder_pre_out_np) * 1/2)**2)
            feature_layer = torch.from_numpy(feature_layer_np).to(device = self.device)
        elif activation == 'bernoulli':
            feature_layer = torch.distributions.bernoulli.Bernoulli(encoder_pre_out).sample() * 2 -1
            regul_loss = 0
        else:
            raise Exception('final_activation_str: {} does not match any of the cases'.format(\
                'self.final_activation_str'))

        return feature_layer, regul_loss

    def convert(self,x,activation = None):

        encoder_pre_out = self.encode(x)
        feature_layer, regul_loss = self.final_activation(encoder_pre_out, activation)

        return feature_layer, regul_loss

    def forward(self, x, activation = None, hard = None):

        feature_layer, regul_loss = self.convert(x, activation)

        res = self.decode(feature_layer)

        return res, feature_layer, regul_loss

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

class VarGenerator(nn.Module):
    def __init__(self, opt, buildModule = True):

        super(VarGenerator, self).__init__()

        self.num_channels = opt.num_channels
        self.num_fc = 1
        self.categorical_dim = opt.categorical_dim
        self.fl_hidden_shape = (opt.batch_size, opt.feature_layer_size, opt.categorical_dim)
        self.fl_flat_shape = (opt.batch_size, opt.feature_layer_size * opt.categorical_dim)
        # print("self.fl_flat_shape",self.fl_flat_shape)
        self.input_shape = (opt.batch_size, opt.input_nc, opt.image_height, opt.image_width)
        self.feature_layer_size = opt.feature_layer_size
        self.use_bias = opt.use_bias
        self.opt = opt

    def __str__(self):
        old = super(VarGenerator, self).__str__()

        self.num_params = 0
        params_str = ''
        for i,param in enumerate(list(self.parameters())):
            params_str = params_str + \
                        '\n\t param: {} - shape: {} - requires_grad: {}'.format(i,param.shape, param.requires_grad)
            self.num_params += param.numel()

        # name = 'MODEL CLASS: {}'.format(self.__class__.__name__)
        add = '\n input shape: {}'.format(self.input_shape) + \
              '\n feature layer shape: {}'.format(self.fl_flat_shape) + \
              '\n feature layer shape: {}'.format(self.fl_hidden_shape) + \
              '\n num fc: {}'.format(self.num_fc) + \
              '\n feature num_channels: {}'.format(self.num_channels) + \
              '\n use_bias {}'.format(self.use_bias) + \
              '\n model params: nbtensors - {}, nbparams - {}'.format(len(list(self.parameters())), self.num_params)
        add = add + params_str
        return old + add


    def build_module(self):

        raise Exception('Not implemented Function: build_module')

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

class VarEncoder(VarGenerator):
    def __init__(self,opt,buildModule = True):

        super(VarGenerator, self).__init__(opt, buildModule)

        if buildModule:
            self.layer_dict = nn.ModuleDict()
            self.build_module()

    def encode(self,x):
        out = x

        for i in range(self.num_layers_encoder):
            # print('{} - A : {}'.format(i,out.shape))
            if i == self.num_layers_encoder - self.num_fc:
                # if out.shape[-1] != 2:
                #     out = F.adaptive_avg_pool2d(out, 2)  # apply adaptive pooling to make sure output of conv layers is always (2, 2) spacially (helps with comparisons).
                out = out.view(self.conversion_layer_shape_after)
            # print('{} - B : {}'.format(i,out.shape))
            out = self.layer_dict['encode_{}'.format(i)](out)
            out = self.layer_dict['encode_{}_activation'.format(i)](out)
            # print('{} - C : {}'.format(i,out.shape))

            if 'encode_{}_reduction'.format(i) in self.layer_dict:
                out = self.layer_dict['encode_{}_reduction'.format(i)](out)

        feature_layer_prob = out.view(self.fl_flat_shape)

        return feature_layer_prob

class VarDecoder(VarGenerator):
    def __init__(self,opt,buildModule = True, shape_args = None):

        super(VarGenerator, self).__init__(opt, buildModule)
        self.conversion_layer_shape_before = shape_args[0]
        self.conversion_layer_shape_after = shape_args[1]

        if buildModule:
            self.layer_dict = nn.ModuleDict()
            self.build_module()


    def decode(self,x):
        out = x.view(self.fl_flat_shape)

        for i in range(self.num_layers_decoder):
            # print('{} - A : {}'.format(i,out.shape))
            if i == self.num_fc:
                out = out.view(self.conversion_layer_shape_before)
            # print('{} - B : {}'.format(i,out.shape))
            out = self.layer_dict['decode_{}'.format(i)](out)

            out = self.layer_dict['decode_{}_activation'.format(i)](out)

            if 'decode_{}_upsampling'.format(i) in self.layer_dict:
                out = self.layer_dict['decode_{}_upsampling'.format(i)](out)

        res = out.view(self.input_shape)
        return res

class VarAutoencoder(nn.Module):

    def __init__(self, opt, buildModule = True, norm_layer = None):

        super(VarAutoencoder, self).__init__()

        self.num_channels = opt.num_channels
        self.num_fc = 1
        self.categorical_dim = opt.categorical_dim
        self.fl_hidden_shape = (opt.batch_size, opt.feature_layer_size, opt.categorical_dim)
        self.fl_flat_shape = (opt.batch_size, opt.feature_layer_size * opt.categorical_dim)
        # print("self.fl_flat_shape",self.fl_flat_shape)
        self.input_shape = (opt.batch_size, opt.input_nc, opt.image_height, opt.image_width)
        self.feature_layer_size = opt.feature_layer_size
        self.opt = opt
        self.norm_layer = norm_layer
        self.use_dropout_encoder = opt.use_dropout_encoder
        self.use_dropout_decoder = opt.use_dropout_decoder

        if type(self.norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            self.use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            self.use_bias = norm_layer != nn.BatchNorm2d

        if buildModule:
            self.layer_dict = nn.ModuleDict()
            self.build_module()


    def __str__(self):
        old = super(VarAutoencoder, self).__str__()

        self.num_params = 0
        params_str = ''
        for i,param in enumerate(list(self.parameters())):
            params_str = params_str + \
                        '\n\t param: {} - shape: {} - requires_grad: {}'.format(i,param.shape, param.requires_grad)
            self.num_params += param.numel()

        # name = 'MODEL CLASS: {}'.format(self.__class__.__name__)
        add = '\n input shape: {}'.format(self.input_shape) + \
              '\n feature layer shape: {}'.format(self.fl_flat_shape) + \
              '\n feature layer shape: {}'.format(self.fl_hidden_shape) + \
              '\n num fc: {}'.format(self.num_fc) + \
              '\n feature num_channels: {}'.format(self.num_channels) + \
              '\n use_bias {}'.format(self.use_bias) + \
              '\n model params: nbtensors - {}, nbparams - {}'.format(len(list(self.parameters())), self.num_params)
        add = add + params_str
        return old + add


    def build_module(self):

        raise Exception('Not implemented Function: build_module')

    def encode(self,x):
        out = x

        for i in range(self.num_layers_encoder):
            # print('{} - A : {}'.format(i,out.shape))
            if i == self.num_layers_encoder - self.num_fc:
                # if out.shape[-1] != 2:
                #     out = F.adaptive_avg_pool2d(out, 2)  # apply adaptive pooling to make sure output of conv layers is always (2, 2) spacially (helps with comparisons).
                out = out.view(self.conversion_layer_shape_after)
            # print('{} - B : {}'.format(i,out.shape))
            out = self.layer_dict['encode_{}'.format(i)](out)

            if 'encode_{}_norm'.format(i) in self.layer_dict:
                out = self.layer_dict['encode_{}_norm'.format(i)](out)

            out = self.layer_dict['encode_{}_activation'.format(i)](out)

            if 'decode_{}_dropout'.format(i) in self.layer_dict:
                # print('- dropout is computed [training: {}]'.format(self.training))
                # zero_count_before = torch.numel(out[0]) - torch.nonzero(out[0]).size(0)
                out = self.layer_dict['decode_{}_dropout'.format(i)](out)
                # zero_count_after = torch.numel(out[0]) - torch.nonzero(out[0]).size(0)
                # print(zero_count_after - zero_count_before, self.training)
            # print('{} - C : {}'.format(i,out.shape))

            if 'encode_{}_reduction'.format(i) in self.layer_dict:
                out = self.layer_dict['encode_{}_reduction'.format(i)](out)

        feature_layer_prob = out.view(self.fl_hidden_shape)

        return feature_layer_prob

    def decode(self,x):
        out = x.view(self.fl_flat_shape)

        for i in range(self.num_layers_decoder):
            # print('{} - A : {}'.format(i,out.shape))
            if i == self.num_fc:
                out = out.view(self.conversion_layer_shape_before)
            # print('{} - B : {}'.format(i,out.shape))
            out = self.layer_dict['decode_{}'.format(i)](out)

            if 'decode_{}_norm'.format(i) in self.layer_dict:
                out = self.layer_dict['decode_{}_norm'.format(i)](out)

            out = self.layer_dict['decode_{}_activation'.format(i)](out)

            if 'decode_{}_dropout'.format(i) in self.layer_dict:
                # print('- dropout is computed [training: {}]'.format(self.training))
                # zero_count_before = torch.numel(out[0]) - torch.nonzero(out[0]).size(0)
                out = self.layer_dict['decode_{}_dropout'.format(i)](out)
                # zero_count_after = torch.numel(out[0]) - torch.nonzero(out[0]).size(0)
                # print(zero_count_after - zero_count_before, self.training)

            if 'decode_{}_upsampling'.format(i) in self.layer_dict:
                out = self.layer_dict['decode_{}_upsampling'.format(i)](out)

        res = out.view(self.input_shape)
        return res

    # def loss_function(self,recon_x,x,encoder_pre_out):
    #     # rec_loss = nn.MSELoss().to(self.device)(recon_x, x)
    #     try:
    #         if self.loss_func == 'bce':
    #             recons_loss = F.binary_cross_entropy(recon_x, x, size_average=False) / x.shape[0]
    #             other_loss = nn.MSELoss().to(self.device)(recon_x, x)
    #         elif self.loss_func == 'mse':
    #             recons_loss = nn.MSELoss().to(self.device)(recon_x, x)
    #             other_loss = F.binary_cross_entropy(recon_x, x, size_average=False) / x.shape[0]
    #     except Exception as e:
    #         print(recon_x.shape, recon_x.max(), recon_x.min(), recon_x.mean())
    #         print(x.shape, x.max(), x.min(), x.mean())
    #         raise e

    #     eps = torch.tensor(1e-20)

    #     q_y = encoder_pre_out.view(encoder_pre_out.size(0), self.feature_layer_size, self.categorical_dim)
    #     qy = F.softmax(q_y, dim=-1).reshape(*encoder_pre_out.size())
    #     log_ratio = torch.log(qy * self.categorical_dim + eps)
    #     kld_loss_term = torch.sum(qy * log_ratio, dim=-1).mean()

    #     kld_loss = self.beta_kld * kld_loss_term
    #     total_loss = recons_loss + kld_loss

    #     losses = {'total_loss': total_loss.data.cpu().numpy()\
    #             , 'recon_loss': recons_loss.data.cpu().numpy()\
    #             , 'kld_loss': kld_loss.data.cpu().numpy()\
    #             , 'mse_loss': other_loss.data.cpu().numpy()}

    #     return total_loss, losses

    # def model_specific_conversion(self,x):
    #     return x

    # def convert(self,x,activation = None, hard = False,temp = 1):

    #     x = self.model_specific_conversion(x)

    #     feature_layer_prob = self.encode(x)
    #     # # encoder_pre_out = encoder_pre_out.view(self.input_shape[0], -1, self.categorical_dim)
    #     # feature_layer = self.final_activation(encoder_pre_out, activation, hard = hard, temp = temp)

    #     # q = encoder_pre_out
    #     # q_y = feature_layer_prob.view(feature_layer_prob.size(0), self.feature_layer_size, self.categorical_dim)
    #     feature_layer = gumbel_softmax(feature_layer_prob, self.categorical_dim, device= self.device\
    #                                         , hard = hard, temperature = temp)
    #     # print('encoder_pre_out', encoder_pre_out)
    #     # print('feature_layer', feature_layer)
    #     return feature_layer, feature_layer_prob

    # def forward(self, x, activation = None, hard = False, temp = 1):

    #     feature_layer, feature_layer_prob = self.convert(x, activation, hard = hard, temp = temp)

    #     res = self.decode(feature_layer)

    #     return res, feature_layer, feature_layer_prob

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

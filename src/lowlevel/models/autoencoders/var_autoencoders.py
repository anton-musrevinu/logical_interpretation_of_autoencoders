from .autoencoder import VarAutoencoder, VarEncoder, VarDecoder
import torch
import torch.nn as nn
import functools
from ..networks import *

class VarLinAutoEncoder(VarAutoencoder):

    def model_specific_conversion(self,x):
        return x.view(-1, 784)

    def build_module(self):

        self.num_fc = 100

        print('building build_model_basic')

        feature_layer = self.build_encoder()

        # ======================== FEATURE LAYER =================================

        print('----- feature layer : {}'.format(feature_layer.shape))

        self.build_decoder(feature_layer)

    def build_encoder(self):
        out = torch.zeros((self.input_shape)).view(-1, 784)
        layer_idx = -1

        #------------------------- encoder - layer 1 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(784, 512)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)
        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(512, 256)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)
        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(256, self.feature_layer_size * self.categorical_dim)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)
        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_encoder = layer_idx + 1

        return out

    def build_decoder(self,feature_layer):
        layer_idx = -1

        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,feature_layer.shape))
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(self.feature_layer_size * self.categorical_dim,256)
        out = self.layer_dict['decode_{}'.format(layer_idx)](feature_layer)
        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(256,512)
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)
        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(512,784)
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)
        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.Sigmoid()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_decoder = layer_idx + 1

class VarConvEncoder(VarEncoder):
    def build_module(self):
        self.num_fc = 2

        out = torch.zeros((self.input_shape))
        layer_idx = -1

        #------------------------- encoder - layer 1 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        # print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=2)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

        #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        # print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=2)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

                #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        # print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=1)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')


        # #------------------------- encoder - layer 1 ----------------------------

        self.conversion_layer_shape_before = out.shape
        layer_idx = layer_idx + 1

            #------------------------- Conv -> FC shape conversion ------------------
        out = out.view(out.shape[0], -1)
        self.conversion_layer_shape_after = out.shape

        # print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * self.categorical_dim * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)
        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')


        # #------------------------- encoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * self.categorical_dim,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        # print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_encoder = layer_idx + 1

class VarConvDecoder(VarDecoder):
    def build_module(self):
        self.num_fc = 2

        layer_idx = -1
        out = torch.zeros((self.fl_flat_shape))

        #------------------------- decoder - layer 0
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), feature_layer.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=fself.feature_layer_size * self.categorical_dim,  # add a linear layer
                                            out_features=self.feature_layer_size * self.categorical_dim * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

        #------------------------- decoder - layer 1
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.conversion_layer_shape_after[1],
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

        out = out.view(self.conversion_layer_shape_before)


        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        # print('output l_{} before upsampling: '.format(layer_idx), out.shape)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(scale_factor=1)
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=4, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        # print('output l_{} before upsampling: '.format(layer_idx), out.shape)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(scale_factor=2)
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')


        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        # print('output l_{} before upsampling: '.format(layer_idx), out.shape)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(scale_factor=2)
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = self.input_shape[1],
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.Sigmoid()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

        self.num_layers_decoder = layer_idx + 1


class VarConvAutoEncoder(VarAutoencoder):

    def build_module(self):

        self.num_fc = 2

        print('building build_model_basic')

        feature_layer = self.build_encoder()

        # ======================== FEATURE LAYER =================================

        print('------------------------------')
        print('----- feature layer : {}'.format(feature_layer.shape))
        print('------------------------------')

        self.build_decoder(feature_layer)

    def build_encoder(self):
        out = torch.zeros((self.input_shape))
        layer_idx = -1

        #------------------------- encoder - layer 1 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        # print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=2)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

        #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        # print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=2)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

                #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        # print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=1)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')


        # #------------------------- encoder - layer 1 ----------------------------

        self.conversion_layer_shape_before = out.shape
        layer_idx = layer_idx + 1

            #------------------------- Conv -> FC shape conversion ------------------
        out = out.view(out.shape[0], -1)
        self.conversion_layer_shape_after = out.shape

        # print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * self.categorical_dim * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)
        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')


        # #------------------------- encoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * self.categorical_dim,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        # print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_encoder = layer_idx + 1
        return out

    def build_decoder(self,feature_layer):
        layer_idx = -1

        #------------------------- decoder - layer 0
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), feature_layer.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=feature_layer.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * self.categorical_dim * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](feature_layer)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

        #------------------------- decoder - layer 1
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.conversion_layer_shape_after[1],
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

        out = out.view(self.conversion_layer_shape_before)


        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        # print('output l_{} before upsampling: '.format(layer_idx), out.shape)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(scale_factor=1)
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=4, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        # print('output l_{} before upsampling: '.format(layer_idx), out.shape)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(scale_factor=2)
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')


        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        # print('output l_{} before upsampling: '.format(layer_idx), out.shape)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(scale_factor=2)
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        # print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = self.input_shape[1],
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.Sigmoid()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        # print('output l_{}: '.format(layer_idx), out.shape)
        # print('------------------------------')

        self.num_layers_decoder = layer_idx + 1




class VarResNetAutoEncoder(VarAutoencoder):

    def __init__(self, opt):

        super(VarResNetAutoEncoder, self).__init__(opt, buildModule = False)

        self.input_nc = opt.input_nc 
        self.output_nc = self.fl_flat_shape[1]
        self.ngf= opt.ngf
        self.normType = opt.norm
        self.use_dropout = opt.no_dropout
        self.n_blocks = 2
        self.padding_type = 'reflect'

        self.n_downsampling = 4
        
        self.build_module()


    def build_module(self):

        self.num_fc = 2

        print('building build_model_basic')

        feature_layer = self.build_encoder(self.input_nc, self.output_nc, use_dropout = not self.opt.no_dropout)

        # ======================== FEATURE LAYER =================================

        print('------------------------------')
        print('----- feature layer : {}'.format(feature_layer.shape))
        print('------------------------------')

        self.build_decoder(feature_layer, self.output_nc, self.input_nc)

    def build_encoder(self,input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):

        out = torch.zeros((self.input_shape))


        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]


        tmp_model = nn.Sequential(*model)
        tmp_out = tmp_model(out)
        print("tmp_out_shape (beofore downsampling): {}".format(tmp_out.shape))

        n_downsampling = self.n_downsampling
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]


        tmp_model = nn.Sequential(*model)
        tmp_out = tmp_model(out)
        print("tmp_out_shape (after downsampling): {}".format(tmp_out.shape))

        mult = 2 ** n_downsampling
        for i in range(self.n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # for i in range(n_downsampling):  # add upsampling layers
        #     mult = 2 ** (n_downsampling - i)
        #     model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                  kernel_size=3, stride=2,
        #                                  padding=1, output_padding=1,
        #                                  bias=use_bias),
        #               norm_layer(int(ngf * mult / 2)),
        #               nn.ReLU(True)]

        # model += [nn.ReflectionPad2d(3)]



        mult = 2 ** (n_downsampling)
        model += [nn.Conv2d(ngf * mult, output_nc * 2, kernel_size=2, padding=0)]
        model += [nn.ReLU()]

        self.enocder_model = nn.Sequential(*model)

        print("input shape: {}".format(out.shape))
        out = self.enocder_model(out)
        self.conversion_layer_shape_before = out.shape
        print("encoder pre out shape: {}".format(out.shape))
        out = out.view(out.shape[0], -1)
        self.conversion_layer_shape_after = out.shape

        model = [nn.Linear(in_features=out.shape[1],  # add a linear layer
                    out_features=output_nc,
                    bias=self.use_bias)]
        model += [nn.ReLU()]

        self.encoder_fcc = nn.Sequential(*model)
        out = self.encoder_fcc(out)
        print("encoder out shape: {}".format(out.shape))
        return out

    def build_decoder(self,feature_layer, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Linear(in_features=feature_layer.shape[1],  # add a linear layer
                    out_features=2 * feature_layer.shape[1],
                    bias=self.use_bias)]
        model += [nn.ReLU()]

        self.decoder_fcc = nn.Sequential(*model)

        out = torch.zeros((feature_layer.shape))
        out = self.decoder_fcc(out)
        print("decoder pre out shape: {}".format(out.shape))
        out = out.view(self.conversion_layer_shape_before)
        print("decoder pre 2 out shape: {}".format(out.shape))

        # model = [nn.ReflectionPad2d(3),
        #          nn.Conv2d(out.shape[1], ngf, kernel_size=7, padding=0, bias=use_bias),
        #          norm_layer(ngf),
        #          nn.ReLU(True)]
        model = []
        n_downsampling = self.n_downsampling
     
      # mult = 2 ** (n_downsampling)
      #   model += [nn.Conv2d(ngf * mult, output_nc * 2, kernel_size=2, padding=0)]
      #   model += [nn.ReLU()]

        mult = 2 ** (n_downsampling )
        model += [nn.ConvTranspose2d(out.shape[1], ngf * mult,
                                     kernel_size=2,
                                     padding=0, 
                                     bias=use_bias),
                  norm_layer(ngf * mult),
                  nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(self.n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        tmp_model = nn.Sequential(*model)
        tmp_out = tmp_model(out)
        print("tmp_out_shape (before upsampling): {}".format(tmp_out.shape))

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]


        tmp_model = nn.Sequential(*model)
        tmp_out = tmp_model(out)
        print("tmp_out_shape (after apsampling): {}".format(tmp_out.shape))

        # model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=1)]
        model += [nn.Sigmoid()]


        # for i in range(n_downsampling):  # add upsampling layers
        #     mult = 2 ** (n_downsampling - i)
        #     model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                  kernel_size=3, stride=2,
        #                                  padding=1, output_padding=1,
        #                                  bias=use_bias),
        #               norm_layer(int(ngf * mult / 2)),
        #               nn.ReLU(True)]
        # model += [nn.ReflectionPad2d(3)]
        # model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # model += [nn.ReLU()]

        self.decoder_model = nn.Sequential(*model)

        out = self.decoder_model(out)
        print("encoder out shape: {}".format(out.shape))

        assert(out.shape == self.input_shape)

        # model = [nn.Linear(in_features=out.shape[1],  # add a linear layer
        #             out_features=output_nc,
        #             bias=self.use_bias)]
        # model += [nn.ReLU()]

        return out


    def encode(self,x):
        out = x

        out = self.enocder_model(out)
        out = out.view(out.shape[0], -1)
        out = self.encoder_fcc(out)

        feature_layer_prob = out.view(self.fl_hidden_shape)

        return feature_layer_prob

    def decode(self,x):
        out = x.view(self.fl_flat_shape)

        out = self.decoder_fcc(out)
        out = out.view(self.conversion_layer_shape_before)
        out = self.decoder_model(out)

        res = out.view(self.input_shape)
        return res

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        models = [self.enocder_model, self.encoder_fcc, self.decoder_model, self.decoder_fcc]
        for model in models:
            for item in model.children():
                try:
                    item.reset_parameters()
                except:
                    pass
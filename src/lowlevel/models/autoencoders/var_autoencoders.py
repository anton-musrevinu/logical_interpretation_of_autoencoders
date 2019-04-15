from .autoencoder import VarAutoencoder
import torch
import torch.nn as nn

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
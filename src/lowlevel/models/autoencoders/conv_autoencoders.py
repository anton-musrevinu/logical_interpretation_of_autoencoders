from .autoencoder import ConvAutoencoder
import torch
import torch.nn as nn

class ConvAutoencoder_Baisc(ConvAutoencoder):

    def build_module(self):

        print('building build_model_basic')

        feature_layer = self.build_encoder_basic()

        # ======================== FEATURE LAYER =================================

        print('----- feature layer : {}'.format(feature_layer.shape))

        self.build_decoder_basic(feature_layer)

    def build_encoder_basic(self):
        out = torch.zeros((self.input_shape))
        layer_idx = -1

        #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=5, stride=3,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)


        # #------------------------- encoder - layer 1 ----------------------------

        self.conversion_layer_shape_before = out.shape
        layer_idx = layer_idx + 1

            #------------------------- Conv -> FC shape conversion ------------------
        out = out.view(out.shape[0], -1)
        self.conversion_layer_shape_after = out.shape

        print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)

        if self.final_activation_str == 'tanh' or self.final_activation_str == 'heaviside':
            self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Tanh()
        elif self.final_activation_str == 'bernoulli':
            self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Sigmoid()
        else:
            raise Exception('final_activation_str: {} does not match any of the cases'.format(\
                'self.final_activation_str'))

        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out * self.feature_leyer_act_scalar)

        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_encoder = layer_idx + 1
        return out

    def build_decoder_basic(self,feature_layer):
        layer_idx = -1

        #------------------------- decoder - layer 0
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), feature_layer.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=feature_layer.shape[1],  # add a linear layer
                                            out_features=self.conversion_layer_shape_after[1],
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](feature_layer)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        out = out.view(self.conversion_layer_shape_before)


        #------------------------- decoder - layer 1 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = self.input_shape[1],
                                        kernel_size=4, stride=3,
                                        padding=0)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.Tanh()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_decoder = layer_idx + 1

class ConvAutoencoder_2fc(ConvAutoencoder):


    def build_module(self):

        self.num_fc = 2

        print('building build_model_basic')

        feature_layer = self.build_encoder_2fc()

        # ======================== FEATURE LAYER =================================

        print('----- feature layer : {}'.format(feature_layer.shape))

        self.build_decoder_2fc(feature_layer)

    def build_encoder_2fc(self):
        out = torch.zeros((self.input_shape))
        layer_idx = -1

        #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=5, stride=3,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)


        # #------------------------- encoder - layer 1 ----------------------------

        self.conversion_layer_shape_before = out.shape
        layer_idx = layer_idx + 1

            #------------------------- Conv -> FC shape conversion ------------------
        out = out.view(out.shape[0], -1)
        self.conversion_layer_shape_after = out.shape

        print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)
        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)


        # #------------------------- encoder - layer 1 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)

        if self.final_activation_str == 'tanh' or self.final_activation_str == 'heaviside':
            self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Tanh()
        elif self.final_activation_str == 'bernoulli':
            self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Sigmoid()
        else:
            raise Exception('final_activation_str: {} does not match any of the cases'.format(\
                'self.final_activation_str'))

        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out * self.feature_leyer_act_scalar)

        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_encoder = layer_idx + 1
        return out

    def build_decoder_2fc(self,feature_layer):
        layer_idx = -1

        #------------------------- decoder - layer 0
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), feature_layer.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=feature_layer.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](feature_layer)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- decoder - layer 1
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.conversion_layer_shape_after[1],
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        out = out.view(self.conversion_layer_shape_before)


        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = self.input_shape[1],
                                        kernel_size=4, stride=3,
                                        padding=0)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.Tanh()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_decoder = layer_idx + 1

class ConvAutoencoder_reduction_2fc(ConvAutoencoder):

    def build_module(self):

        self.num_fc = 2

        print('building build_model_basic')

        feature_layer = self.build_encoder()

        # ======================== FEATURE LAYER =================================

        print('----- feature layer : {}'.format(feature_layer.shape))

        self.build_decoder(feature_layer)

    def build_encoder(self):
        out = torch.zeros((self.input_shape))
        layer_idx = -1

        #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=5, stride=3,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=1)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)


        # #------------------------- encoder - layer 1 ----------------------------

        self.conversion_layer_shape_before = out.shape
        layer_idx = layer_idx + 1

            #------------------------- Conv -> FC shape conversion ------------------
        out = out.view(out.shape[0], -1)
        self.conversion_layer_shape_after = out.shape

        print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)
        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)


        # #------------------------- encoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)

        if self.final_activation_str == 'tanh' or self.final_activation_str == 'heaviside':
            self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Tanh()
        elif self.final_activation_str == 'bernoulli':
            self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Sigmoid()
        else:
            raise Exception('final_activation_str: {} does not match any of the cases'.format(\
                'self.final_activation_str'))

        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out * self.feature_leyer_act_scalar)

        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_encoder = layer_idx + 1
        return out

    def build_decoder(self,feature_layer):
        layer_idx = -1

        #------------------------- decoder - layer 0
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), feature_layer.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=feature_layer.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](feature_layer)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- decoder - layer 1
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.conversion_layer_shape_after[1],
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        out = out.view(self.conversion_layer_shape_before)


        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = self.num_channels,
                                        kernel_size=2, stride=1,
                                        padding=0)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.Tanh()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = self.input_shape[1],
                                        kernel_size=4, stride=3,
                                        padding=0)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.Tanh()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_decoder = layer_idx + 1

class ConvAutoencoder_full_symetric(ConvAutoencoder):

    def build_module(self):

        self.num_fc = 2

        print('building build_model_basic')

        feature_layer = self.build_encoder()

        # ======================== FEATURE LAYER =================================

        print('----- feature layer : {}'.format(feature_layer.shape))

        self.build_decoder(feature_layer)

    def build_encoder(self):
        out = torch.zeros((self.input_shape))
        layer_idx = -1

        #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=2)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels/2),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=2)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)


        # #------------------------- encoder - layer 1 ----------------------------

        self.conversion_layer_shape_before = out.shape
        layer_idx = layer_idx + 1

            #------------------------- Conv -> FC shape conversion ------------------
        out = out.view(out.shape[0], -1)
        self.conversion_layer_shape_after = out.shape

        print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)
        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)


        # #------------------------- encoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)

        if self.final_activation_str == 'tanh' or self.final_activation_str == 'heaviside':
            self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Tanh()
        elif self.final_activation_str == 'bernoulli':
            self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Sigmoid()
        else:
            raise Exception('final_activation_str: {} does not match any of the cases'.format(\
                'self.final_activation_str'))

        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out * self.feature_leyer_act_scalar)

        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_encoder = layer_idx + 1
        return out

    def build_decoder(self,feature_layer):
        layer_idx = -1

        #------------------------- decoder - layer 0
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), feature_layer.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=feature_layer.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](feature_layer)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- decoder - layer 1
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.conversion_layer_shape_after[1],
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        out = out.view(self.conversion_layer_shape_before)


        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels/2),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(scale_factor=2)
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)


        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(scale_factor=2)
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = self.input_shape[1],
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.Tanh()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_decoder = layer_idx + 1

class ConvAutoencoder_full_symetric_conv(ConvAutoencoder):

    def build_module(self):

        self.num_fc = 2

        print('building build_model_basic')

        feature_layer = self.build_encoder()

        # ======================== FEATURE LAYER =================================

        print('----- feature layer : {}'.format(feature_layer.shape))

        self.build_decoder(feature_layer)

    def build_encoder(self):
        out = torch.zeros((self.input_shape))
        layer_idx = -1

        #------------------------- encoder - layer 1 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=2)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=2)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)

                #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=4, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=2)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)


        # #------------------------- encoder - layer 1 ----------------------------

        self.conversion_layer_shape_before = out.shape
        layer_idx = layer_idx + 1

            #------------------------- Conv -> FC shape conversion ------------------
        out = out.view(out.shape[0], -1)
        self.conversion_layer_shape_after = out.shape

        print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)
        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)


        # #------------------------- encoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)

        if self.final_activation_str == 'tanh' or self.final_activation_str == 'heaviside':
            self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Tanh()
        elif self.final_activation_str == 'bernoulli' or self.final_activation_str == 'gumbell':
            self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Sigmoid()
        else:
            raise Exception('final_activation_str: {} does not match any of the cases'.format(\
                'self.final_activation_str'))

        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out * self.feature_leyer_act_scalar)

        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_encoder = layer_idx + 1
        return out

    def build_decoder(self,feature_layer):
        layer_idx = -1

        #------------------------- decoder - layer 0
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), feature_layer.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=feature_layer.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](feature_layer)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- decoder - layer 1
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.conversion_layer_shape_after[1],
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        out = out.view(self.conversion_layer_shape_before)


        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(scale_factor=2)
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=4, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(scale_factor=2)
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)


        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(scale_factor=2)
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = self.input_shape[1],
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.Tanh()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_decoder = layer_idx + 1

class ConvAutoencoder_full_symetric_deep(ConvAutoencoder):

    def build_module(self):

        self.num_fc = 2

        print('building build_model_basic')

        feature_layer = self.build_encoder()

        # ======================== FEATURE LAYER =================================

        print('----- feature layer : {}'.format(feature_layer.shape))

        self.build_decoder(feature_layer)

    def build_encoder(self):
        out = torch.zeros((self.input_shape))
        layer_idx = -1

        #------------------------- encoder - layer 1 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=5, stride=1,
                                        padding=2)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(3, stride=2)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)

                #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)

                #------------------------- encoder - layer 0 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: {}'.format(layer_idx,out.shape))
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        print('output l_{} before reduction: '.format(layer_idx), out.shape)

        self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(3, stride=2)
        out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)


        # #------------------------- encoder - layer 1 ----------------------------

        self.conversion_layer_shape_before = out.shape
        layer_idx = layer_idx + 1

            #------------------------- Conv -> FC shape conversion ------------------
        out = out.view(out.shape[0], -1)
        self.conversion_layer_shape_after = out.shape

        print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)

        self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

        if self.dropout:
            print('output l_{} before reduction: '.format(layer_idx), out.shape)
            self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.Dropout(.5)
            out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)


        # #------------------------- encoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
        self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size,
                                            bias=self.use_bias)
        out = self.layer_dict['encode_{}'.format(layer_idx)](out)

        if self.final_activation_str == 'tanh' or self.final_activation_str == 'heaviside':
            self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Tanh()
        elif self.final_activation_str == 'bernoulli':
            self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Sigmoid()
        else:
            raise Exception('final_activation_str: {} does not match any of the cases'.format(\
                'self.final_activation_str'))

        out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out * self.feature_leyer_act_scalar)

        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_encoder = layer_idx + 1
        return out

    def build_decoder(self,feature_layer):
        layer_idx = -1

        #------------------------- decoder - layer 0
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), feature_layer.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=feature_layer.shape[1],  # add a linear layer
                                            out_features=self.feature_layer_size * 2,
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](feature_layer)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- decoder - layer 1
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.conversion_layer_shape_after[1],
                                            bias=self.use_bias)
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        if self.dropout:
            self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.Dropout(.3)
            out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)

        print('output l_{}: '.format(layer_idx), out.shape)

        out = out.view(self.conversion_layer_shape_before)


        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(size=(13,13))
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- decoder - layer 3 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- decoder - layer 4 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)


        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = int(self.num_channels),
                                        kernel_size=5, stride=1,
                                        padding=2)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)

        print('output l_{}: before upsampling'.format(layer_idx), out.shape)

        self.layer_dict['decode_{}_upsampling'.format(layer_idx)] = nn.UpsamplingBilinear2d(size = (28,28))
        out = self.layer_dict['decode_{}_upsampling'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        #------------------------- decoder - layer 2 ----------------------------
        layer_idx = layer_idx + 1
        print('input l_{}: '.format(layer_idx), out.shape)
        self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
                                        out_channels = self.input_shape[1],
                                        kernel_size=3, stride=1,
                                        padding=1)  # b, 16, 10, 10
        out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

        self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.Tanh()
        out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
        print('output l_{}: '.format(layer_idx), out.shape)

        self.num_layers_decoder = layer_idx + 1





# def build_encoder(self):
#         out = torch.zeros((self.input_shape))
#         layer_idx = -1

#         #------------------------- encoder - layer 0 ----------------------------
#         layer_idx = layer_idx + 1
#         print('input l_{}: {}'.format(layer_idx,out.shape))
#         self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
#                                         out_channels = int(self.num_channels),
#                                         kernel_size=5, stride=3,
#                                         padding=1)  # b, 16, 10, 10
#         out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

#         self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
#         out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)

#         # self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=1)
#         # out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)
#         print('output l_{}: '.format(layer_idx), out.shape)


#         # #------------------------- encoder - layer 1 ----------------------------
#         # layer_idx = layer_idx + 1
#         # print('input l_{}: {}'.format(layer_idx,out.shape))
#         # self.layer_dict['encode_{}'.format(layer_idx)] = nn.Conv2d(in_channels = out.shape[1],
#         #                                 out_channels = self.num_channels,
#         #                                 kernel_size=3, stride=2,
#         #                                 padding=1)  # b, 16, 10, 10
#         # out = self.layer_dict['encode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

#         # self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.ReLU()
#         # out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out)


#         # # self.layer_dict['encode_{}_reduction'.format(layer_idx)] = nn.MaxPool2d(2, stride=1)
#         # # out = self.layer_dict['encode_{}_reduction'.format(layer_idx)](out)

#         # print('output l_{}: '.format(layer_idx), out.shape)

#         #------------------------- Conv -> FC shape conversion ------------------
#         # if out.shape[-1] != 2:
#         #     out = F.adaptive_avg_pool2d(out, 2)  # apply adaptive pooling to make sure output of conv layers is always (2, 2) spacially (helps with comparisons).


#         #------------------------- encoder - layer 2 ----------------------------

#         self.conversion_layer_shape_before = out.shape
#         layer_idx = layer_idx + 1

#             #------------------------- Conv -> FC shape conversion ------------------
#         out = out.view(out.shape[0], -1)
#         self.conversion_layer_shape_after = out.shape

#         print('input l_{}: '.format(layer_idx), out.shape, out.shape, out.shape[1])
#         self.layer_dict['encode_{}'.format(layer_idx)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
#                                             out_features=self.feature_layer_size,
#                                             bias=self.use_bias)
#         out = self.layer_dict['encode_{}'.format(layer_idx)](out)


#         self.layer_dict['encode_{}_activation'.format(layer_idx)] = nn.Sigmoid()
#         out = self.layer_dict['encode_{}_activation'.format(layer_idx)](out * self.feature_leyer_act_scalar)

#         print('output l_{}: '.format(layer_idx), out.shape)

#         self.num_layers_encoder = layer_idx + 1
#         return ou

# def build_decoder(self,feature_layer):
#     layer_idx = -1

#     #------------------------- decoder - layer 0
#     layer_idx = layer_idx + 1
#     print('input l_{}: '.format(layer_idx), feature_layer.shape)
#     self.layer_dict['decode_{}'.format(layer_idx)] = nn.Linear(in_features=feature_layer.shape[1],  # add a linear layer
#                                         out_features=self.conversion_layer_shape_after[1],
#                                         bias=self.use_bias)
#     out = self.layer_dict['decode_{}'.format(layer_idx)](feature_layer)

#     self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
#     out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
#     print('output l_{}: '.format(layer_idx), out.shape)

#     out = out.view(self.conversion_layer_shape_before)


#     #------------------------- decoder - layer 1 ----------------------------


#         #------------------------- Conv -> FC shape conversion ------------------

#     # layer_idx = layer_idx + 1
#     # print('input l_{}: '.format(layer_idx), out.shape)
#     # self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = int(self.num_channels),
#     #                                 out_channels = self.num_channels,
#     #                                 kernel_size=4, stride=2, padding = 1)  # b, 16, 10, 10
#     # out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

#     # self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
#     # out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
#     # print('output l_{}: '.format(layer_idx), out.shape)


#     #------------------------- decoder - layer 2 ----------------------------
#     # layer_idx = layer_idx + 1
#     # print('input l_{}: '.format(layer_idx), out.shape)
#     # self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = self.num_channels,
#     #                                 out_channels = int(self.num_channels/2),
#     #                                 kernel_size=5, stride=2)  # b, 16, 10, 10
#     # out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

#     # self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.ReLU()
#     # out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
#     # print('output l_{}: '.format(layer_idx), out.shape)

#     #------------------------- decoder - layer 3 ----------------------------

#     layer_idx = layer_idx + 1
#     print('input l_{}: '.format(layer_idx), out.shape)
#     self.layer_dict['decode_{}'.format(layer_idx)] = nn.ConvTranspose2d(in_channels = out.shape[1],
#                                     out_channels = self.input_shape[1],
#                                     kernel_size=4, stride=3,
#                                     padding=0)  # b, 16, 10, 10
#     out = self.layer_dict['decode_{}'.format(layer_idx)](out)  # use layer on inputs to get an output

#     self.layer_dict['decode_{}_activation'.format(layer_idx)] = nn.Tanh()
#     out = self.layer_dict['decode_{}_activation'.format(layer_idx)](out)
#     print('output l_{}: '.format(layer_idx), out.shape)

#     self.num_layers_decoder = layer_idx + 1

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_ae as networks
from collections import OrderedDict
from options import str2bool
import torch.nn.functional as F
import numpy as np


class VAEModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
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
        # parser.set_defaults(no_dropout=True)
        # if is_train:
        #     parser.add_argument('--critic_iter', type=int, default=5, help='number of iterations the discriminater is trained for each training update for the generator')
        #     parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        #     parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        #     parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        
        parser.add_argument('--beta_kld', nargs="?",type=float, default=.7)
        parser.add_argument('--hard', nargs="?",type=str2bool, default=True)
        parser.add_argument('--loss_type',type=str, default='bce')
        parser.add_argument('--num_channels', nargs="?",type=int, default=64)
        parser.add_argument('--use_bias', nargs="?",type=str2bool, default=True)

        parser.add_argument('--ae_model_type', type=str, default='vanilla') #Other option is 'resnet'

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['BCE', 'KLD','MSE']
        self.acc_names = []
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>

       # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['AE']
        else:  # during test time, only load Gs
            self.model_names = ['AE']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        if self.opt.ae_model_type == 'vanilla':
            self.netAE = networks.define_AE(opt)
        else:
            self.netAE = networks.define_resNetAE(opt)

        if self.isTrain:
            # define loss functions

            # self.criterionBCE = torch.nn.BCELoss().to(self.device)
            # self.criterionBCE = lambda x, y: F.binary_cross_entropy(x, y, size_average=False) / x.shape[0
            # self.criterionBCE = torch.nn.L1Loss().to(self.device)
            self.criterionBCE = torch.nn.BCEWithLogitsLoss(self.device)
            self.criterionMSE = torch.nn.MSELoss().to(self.device)
            self.criterionGumbell = networks.Gumbell_kld(opt.beta_kld,opt.categorical_dim, self.netAE.fl_flat_shape).to(self.device)
            self.optimizer = torch.optim.Adam(self.netAE.parameters(), lr=opt.lr, amsgrad=False, weight_decay=opt.weight_decay_coefficient)#, lr=opt.lr)
            # self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

            self.annealing_temp = 1

        # try:
        #     if self.loss_func == 'bce':
        #         recons_loss = F.binary_cross_entropy(recon_x, x, size_average=False) / x.shape[0]
        #         other_loss = nn.MSELoss().to(self.device)(recon_x, x)
        #     elif self.loss_func == 'mse':
        #         recons_loss = nn.MSELoss().to(self.device)(recon_x, x)
        #         other_loss = F.binary_cross_entropy(recon_x, x, size_average=False) / x.shape[0]
        # except Exception as e:
        #     print(recon_x.shape, recon_x.max(), recon_x.min(), recon_x.mean())
        #     print(x.shape, x.max(), x.min(), x.mean())
        #     raise e

        # eps = torch.tensor(1e-20)

        # q_y = encoder_pre_out.view(encoder_pre_out.size(0), self.feature_layer_size, self.categorical_dim)
        # qy = F.softmax(q_y, dim=-1).reshape(*encoder_pre_out.size())
        # log_ratio = torch.log(qy * self.categorical_dim + eps)
        # kld_loss_term = torch.sum(qy * log_ratio, dim=-1).mean()

        # kld_loss = self.beta_kld * kld_loss_term
        # total_loss = recons_loss + kld_loss


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # AtoB = self.opt.direction == 'AtoB'

        #Standard is A: USPS
        #            B: MNSIT
        self.input = input['inputs'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
    def set_fl(self,fl):
        self.feature_layer = fl.to(self.device)

    def get_fl_as_img(self):

        # print(self.feature_layer[0])
        # print(self.feature_layer_prob[0])
        # print(self.feature_layer.shape, self.feature_layer_prob.shape)
        # tmp = torch.argmax(self.feature_layer.clone().detach(), dim = -1)
        # devider = torch.tensor(self.opt.categorical_dim - 1, requires_grad=False).to(self.device)
        # print('tmp: {}, elem: {}, type: {}, devider: {}, devider type: {}'.format(tmp.shape, tmp[0:1], tmp.dtype, devider, devider.dtype))
        fl_discrete = torch.argmax(self.feature_layer, dim = -1).float()
        devider = torch.tensor(1.0/(self.opt.categorical_dim - 1), requires_grad = False).float().to(self.device)

        # print('tmp: {}, elem: {}, type: {}, devider: {}, devider type: {}'.format(fl_discrete.shape, fl_discrete[0:1], 
                # fl_discrete.dtype, devider, devider.dtype))
        fl_discrete_norm = torch.mul(fl_discrete, devider) #0/4 - 0, 1/4 - .25, 2/4 - .5, 3/4 -0.75
        # print('fl_as_discrete:', fl_discrete_norm.shape, fl_discrete_norm[0:1], fl_discrete_norm.dtype)
        diff = int(self.opt.image_height - fl_discrete_norm.shape[1] % self.opt.image_height)

        if diff >= 0:
            fill = torch.zeros(self.opt.batch_size, diff, dtype = torch.float).to(self.device)
            fl_as_img = torch.cat((fl_discrete_norm, fill), 1)
            # print('new shape', fl_as_img.shape, 'old shap', fl_discrete_norm.shape)
            # print(fl_as_img[:,-diff:])
        else:
            fl_as_img = fl_discrete_norm

        # fl_as_img = torch.cat((torch.ones(self.opt.batch_size,2, dtype = torch.long), fl_as_img,torch.ones(self.opt.batch_size,2, dtype = torch.long)), 1)
        # print('padded fl is: {}'.format(fl_as_img.shape))
        fl_as_img = fl_as_img.view(self.opt.batch_size, 1, self.opt.image_height, -1)
        # print(fl_as_img.clone().view(self.opt.batch_size, -1)[:,-diff:])
        # print(fl_as_img[:,1,self.opt.image_height,:])

        x_padding_white = torch.ones((self.opt.batch_size,1,self.opt.image_height, 2), dtype = torch.float).to(self.device)
        x_padding_black = torch.zeros((self.opt.batch_size,1,self.opt.image_height, 2), dtype = torch.float).to(self.device)

        # print('fl without padding: {}'.format(fl_as_img.shape))
        fl_as_img = torch.cat((x_padding_black,x_padding_white,x_padding_black, fl_as_img), 3)
        fl_as_img = torch.cat((fl_as_img, x_padding_black,x_padding_white,x_padding_black), 3)
        # print('fl with padding: {}'.format(fl_as_img.shape))


        return fl_as_img

    def run_encoder(self,training = False):

        self.feature_layer_prob = self.netAE.encode(self.input)
            #Returns feature_layer_prob as tesor BSxFLSxCD

        hard = not training or self.opt.hard
        self.feature_layer = networks.gumbel_softmax(self.feature_layer_prob, 
                        device=self.device, hard = hard, temperature = self.annealing_temp)

    def run_decoder(self):
        self.rec_input = self.netAE.decode(self.feature_layer)

    def forward(self, training = False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.run_encoder(training)
        self.run_decoder()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.compute_network(training = True)

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        self.loss.backward()  # backpropagate to compute gradients for current iter loss
        self.optimizer.step()  # update network parameters

    def compute_network(self, training = False):
        self.forward(training)
        self.loss_BCE = self.criterionBCE(self.rec_input, self.input)  # compute loss
        self.loss_KLD = self.criterionGumbell(self.feature_layer_prob)
        self.loss_MSE = self.criterionMSE(self.rec_input, self.input)
        if self.opt.loss_type == 'bce':
            self.loss = (self.loss_BCE + self.loss_KLD)
        elif self.opt.loss_type == 'mse':
            self.loss = (self.loss_MSE + self.loss_KLD)

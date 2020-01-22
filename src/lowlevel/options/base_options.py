import argparse
import os
from util import util
import torch
import models
import managers
import data
import shutil
import time
from . import str2bool
import importlib

ROOT_LOWLEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), './../'))
ROOTDIR = os.path.abspath(os.path.join(ROOT_LOWLEVEL_DIR, './../..'))

DEFAULT_DATAST_DIR = os.path.abspath(os.path.join(ROOTDIR, './datasets'))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(ROOTDIR, './output'))

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', default=DEFAULT_DATAST_DIR, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        parser.add_argument('--experiment_name', type=str, default='test', help='everything is saved here')
        parser.add_argument('--testing', type=str2bool, default=False, help='reduce epoch and batch size for testing')
        parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='everything is saved here')     
        # model parameters
        parser.add_argument('--dataset', type=str, default='sln', help='chooses dataset to use')

       # additional parameters
        # parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', type=str2bool, default=False, help='if specified, print more debugging information')
        # EXPERIMENT parameters
        parser.add_argument('--phase', required = True, help='specifcy the kind of program you want to run: curretnly implemned: {}'.format(['train, genData']))

        self.initialized = True
        return parser

    def load_base_option(self,parser):   

        parser.add_argument('--model', type=str, default='vae', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization, classifier]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--image_width', type = int)
        parser.add_argument('--image_height', type = int)

        # dataset parameters
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--num_batches', type=int, default=-1)

        return parser

    def load_options_from_file(self,opt, opt_specific):
        opt.loading = self.isTrain and opt.epoch_count != 1 or not self.isTrain
        if opt.loading:
            experiment_dir = os.path.abspath(opt.output_dir + '/experiments/' + opt.experiment_name)
            opt_file = os.path.join(experiment_dir, 'opt.txt')

            if opt.loading and not os.path.exists(opt_file):
                raise Exception('Specified load exeriment; but opt file does not exist: {}'.format(opt_file))
            
            args_as_str = list([])
            with open(opt_file,'r') as f:
                for line in f:
                    if ':' in line:
                        option_pair = line.strip().split(':')
                        option = option_pair[0]
                        value = option_pair[1].split('[')[0].strip()
                        if not hasattr(opt, option):
                            args_as_str.append('--{}={}'.format(option, value))
        else:
            for k,v in vars(opt_specific).items():
                args_as_str.append('--{}={}'.format(k, v))

        for k,v in vars(opt).items():
            args_as_str.append('--{}={}'.format(k, v))

        return args_as_str

    def get_all_specific_options_from_file(self, opt):
        opt.loading = self.isTrain and opt.epoch_count != 1 or not self.isTrain
        args_as_dic = []
        if opt.loading:
            experiment_dir = os.path.abspath(opt.output_dir + '/experiments/' + opt.experiment_name)
            opt_file = os.path.join(experiment_dir, 'opt.txt')

            if opt.loading and not os.path.exists(opt_file):
                raise Exception('Specified load exeriment; but opt file does not exist: {}'.format(opt_file))
            
            with open(opt_file,'r') as f:
                for line in f:
                    if ':' in line:
                        option_pair = line.strip().split(':')
                        option = option_pair[0]
                        value = option_pair[1].split('[')[0].strip()
                        if not hasattr(opt, option) or option == 'dataset':
                            args_as_dic.append('--{}={}'.format(option,value))

        # print('\n\ndict without main',args_as_dic)
        for kk, vv in vars(opt).items():
            if str(kk) == 'dataset':
                continue
            args_as_dic.append('--{}={}'.format(kk,vv))


        # print('\n\ndict with main',args_as_dic)

        return args_as_dic, opt.loading

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        #Phase Specific option:
        method_name = "phase_" + opt.phase + "_options"
        module = importlib.import_module('options.phase_specific_options')
        method = getattr(module, method_name)
        parser = method(parser)
        opt, _ = parser.parse_known_args()
        self.isTrain = opt.phase == 'train'

        print('so far', opt)
        # print('str_options_from_file',str_options_from_file)

        dict_options, isLoading = self.get_all_specific_options_from_file(opt)

        print('so far 2', dict_options)

        # parser_phase_sepcific = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.load_base_option(parser)

        # if isLoading:
        # print('\n\nis loading is true: {}'.format(dict_options))
        opt_spec, _ = parser.parse_known_args(dict_options)


        # print('\n\nso far so good')

        managers_option_setter = managers.get_option_setter(opt_spec.model)
        parser = managers_option_setter(parser, self.isTrain)
        # opt, _ = parser.parse_known_args()  # parse again with new defaults

        model_option_setter = models.get_option_setter(opt_spec.model)
        parser = model_option_setter(parser, self.isTrain)
        # opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_option_setter = data.get_option_setter(opt_spec.dataset)
        print(dataset_option_setter)
        parser = dataset_option_setter(parser, self.isTrain)
        # opt, _ = parser.parse_known_args()

        # parser_all = method(parser_phase_sepcific)
        # parser_all = self.initialize(parser_all)
        self.parser = parser


        # print('so far so good - last')

        options = parser.parse_args()
        # print(options)

        # print('so far so good - last last')
        # print('\n\n\n',options)
        # print('\n\n\n',options2)

        # print('so far so good')
        # todelete = []
        # for idx,elem in enumerate(dict_options):
        #     kk = elem.replace('--','').split('=')[0]
        #     if not hasattr(options2, kk):
        #         todelete.append(idx)
        # print('deleting elements: {}'.format(todelete))
        # for i in todelete:
        #     del dict_options[i]
        if isLoading:
            options2 = parser.parse_known_args(dict_options)
            # print('so far so good - last last last')
            for kk, vv in vars(options2[0]).items():
                if hasattr(options, kk) and getattr(options, kk) != vv:
                    old = getattr(options, kk)
                    setattr(options, kk, vv)
                    print('updating option: {} from {} to {}'.format(kk,old, vv))
                    
        return options


        # save and return the parser
        # self.parser = parser
        # print('str_options_from_file',str_options_from_file, len(str_options_from_file), type(str_options_from_file))
        # if str_options_from_file == []:
        #     print('doing this')
        #     return parser.parse_args()
        # else:
        #     print('doing that')
        #     opt1, _ = parser.parse_known_args()
        #     print('opt1 parsed')
        #     opt2, _ = parser_phase_sepcific.parse_known_args()
        #     opt3, _ = parser_phase_sepcific.parse_known_args(str_options_from_file)
        #     for kk,vv in vars(opt2).items():
        #         vv1 = getattr(op1, kk)
        #         if vv1 != vv:
        #             print('updating option: {} from {} to {}'.format(kk, vv1, vv))
        #         vv1 = vv
        #     return opt2

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.abspath(opt.experiment_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        if not os.path.exists(file_name):
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def make_experiment_dir(self,opt):
        experiment_name = opt.experiment_name
        experiment_path =  opt.output_dir + '/experiments/'
        new_experiment_folder = os.path.abspath(experiment_path + experiment_name)

        opt.loading = self.isTrain and opt.epoch_count != 1 or not self.isTrain
        if opt.loading and not os.path.exists(new_experiment_folder):
            raise Exception('Specified load exeriment; but experiemn folder does not exist: {}'.format(new_experiment_folder))

        if not opt.loading:
            print('a new experiment, is started')
            if not opt.replace_existing:
                this_experiment_counter = 0
                tmp_experiment_name = experiment_name + '_{}'.format(this_experiment_counter)
                tmp_experiment_folder = os.path.abspath(experiment_path + tmp_experiment_name)
                while os.path.exists(tmp_experiment_folder):
                    this_experiment_counter += 1
                    tmp_experiment_name = experiment_name + '_{}'.format(this_experiment_counter)
                    tmp_experiment_folder = os.path.abspath(experiment_path + tmp_experiment_name)
                experiment_name = tmp_experiment_name
                new_experiment_folder = tmp_experiment_folder
            elif os.path.exists(new_experiment_folder):
                print('replaceing existing tree: {}'.format(new_experiment_folder))
                shutil.rmtree(new_experiment_folder)
                time.sleep(2)
            os.makedirs(new_experiment_folder)
        else:
            print('A network is being loaded')

        opt.experiment_dir = new_experiment_folder

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        # print('so far so good')
        self.make_experiment_dir(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])


        if opt.testing and self.isTrain:
            print('testing is true, setting: {}')
            opt.batch_size = 20
            opt.num_epochs = 2
            print('\topt.batch_size = {}'.format(opt.batch_size))
            print('\topt.num_epoch = {}'.format(opt.num_epochs))
        elif opt.experiment_name == 'test' and self.isTrain:
            raise Exception('experiment_name is required when not testing')


        self.print_options(opt)

        self.opt = opt
        return self.opt

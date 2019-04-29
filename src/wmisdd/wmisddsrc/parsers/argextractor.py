import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2tupefloat(v,type):
    l = []
    try:
        for e in v.strip().split(','):
            l.append(type(e))
    except Exception:
        raise argparse.ArgumentTypeError('List of float values expected. [\'x,y,z\']')
    return tuple(l)

def str2liststr(v,type):
    l = []
    try:
        for e in v.strip().split(','):
            l.append(e)
    except Exception:
        raise argparse.ArgumentTypeError('List of float values expected. [\'x,y,z\']')
    return l

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the WMISDD python Libary')

    parser.add_argument('--mode', nargs="?", type=str, default='WMI', help='Select the mode you want to enter: \
                                                                           [WMI,SDD] ')

    parser.add_argument('--logger', nargs="?", type=str, default='console', help='Select the logger you want: \
                                                                           [file (must specify file location with ,\
                                                                            console (default)] ')
    parser.add_argument('--logger_file', nargs="?", type=str, default=None, help='Specify logger file')

    parser.add_argument('--name', nargs="?", type=str, default='exp_0', help='Name of the experiment')

    parser.add_argument('--tmpdir', nargs="?", type=str, default=None, help='Path to tmp dir')

    parser.add_argument('--interror', nargs="?", type=str2tupefloat, default=(1,1), help='Integration Error for Latte')

    parser.add_argument('--kbfile', nargs="?", type=str, default=None, help='Path to kbfile dir')

    parser.add_argument('--wffile', nargs="?", type=str, default=None, help='Path to wffile dir')

    parser.add_argument('--integration_method', nargs="?", type=str, default='scipy', help='the integration library used: [latte,scipy]')

    parser.add_argument('--onehot_numvars', nargs="?", type=int, default=-1, help='number of variables in the feature layer')

    parser.add_argument('--cnf_dir', type =str, default = None)
    parser.add_argument('--onehot_fl_size', nargs="?", type=int, default=-1, help='number of variables in the x layer (not to be one hot encoded)')
    parser.add_argument('--onehot_fl_categorical_dim', type=int, default=1, help='number of categorical dimention of the feature layers')
    parser.add_argument('--onehot_out_sdd', nargs="?", type=str, default=None, help='output path of the sdd file (not provided will write it to the tmp file)')
    parser.add_argument('--onehot_out_vtree', nargs="?", type=str, default=None, help='output path of the vtree file (not provided will write it to the tmp file)')
    parser.add_argument('--precomputed_vtree', type=str2bool, default = False)

    parser.add_argument('--keeptmpdir', nargs="?", type=str2bool, default=False, help='specify if the tmpdir should be deleted after the execution finished')

    parser.add_argument('--wme_in_sdd', nargs="?", type=str, default=None, help='output path of the sdd file (not provided will write it to the tmp file)')
    parser.add_argument('--wme_in_vtree', nargs="?", type=str, default=None, help='output path of the vtree file (not provided will write it to the tmp file)')
    

    args = parser.parse_args()
    # print(args)
    return args

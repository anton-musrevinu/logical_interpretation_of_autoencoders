from . import str2bool

def phase_train_options(parser):


    parser.add_argument('--replace_existing', type=str2bool, default=True, help='replace the previously created experiment if present')

    parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--continue_train', type=str2bool, default=False, help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=1e-05, help='Weight decay to use for Adam')

    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', type=str2bool, default=True, help='no dropout for the generator')


    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='The experiment\'s epoch budget')
    return parser

def phase_graph_options(parser):

    return parser


def phase_encode_options(parser):
    parser.add_argument('--for_error', type=str, default='mse', help='picking the best model for the given error on the validation set')
    parser.add_argument('--limit_conversion', type=int, default=-1, help='the maximum size of the datset the is bening created')
    parser.add_argument('--compress_fly', type=str2bool, default=True, help='Compressing fly to a binary represenations enforcing onehot')
    return parser

def phase_decode_options(parser):
    parser.add_argument('--for_error', type=str, default='mse', help='picking the best model for the given error on the validation set')
    parser.add_argument('--file_to_decode', type=str)
    return parser


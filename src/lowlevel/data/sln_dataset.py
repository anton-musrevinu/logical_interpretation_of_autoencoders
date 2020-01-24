from data.image_folder import ImageDataset

class SLNDataset(ImageDataset):
	"""Data provider for MNIST handwritten digit images."""


	@staticmethod
	def modify_commandline_options(parser, is_train):
		"""Add new dataset-specific options, and rewrite default values for existing options.

		Parameters:
			parser          -- original option parser
			is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.

		By default, the number of channels for input image  is 1 (L) and
		the nubmer of channels for output image is 2 (ab). The direction is from A to B
		"""
		parser.add_argument('--num_classes', type=int, default=10)
		parser.set_defaults(input_nc=3, batch_size=50, image_width = 96, image_height = 96)
		return parser

	def __init__(self, opt, type_of_data, mydir = None):
		# print('initializing the right dataset')
		if opt.phase == 'train':
			domain = 'sln-unsupervised'
		else:
			domain = 'sln-supervised'
		ImageDataset.__init__(self, opt, domain, type_of_data, mydir)

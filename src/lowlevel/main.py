#Main module, used to control the rest of the system by starting and building the experiment.

import managers
from options.base_options import BaseOptions

# def train_autoencoder(opt):

# 	managerClass = managers.find_manager_using_name(opt.model)
# 	manager = managerClass(opt)
# 	manager.do_training()

# def conver_dataset(opt):
# 	manager.print_info()

if __name__ == '__main__':
	opt = BaseOptions().parse()

	managerClass = managers.find_manager_using_name(opt.model)
	manager = managerClass(opt)

	if opt.phase == 'train':
		manager.do_training()
	elif opt.phase == 'info':
		manager.print_info()
	elif opt.phase == 'encode':
		manager.convert_all_data()
	elif opt.phase == 'decode':
		manager.decode_specific_file(opt.file_to_decode)

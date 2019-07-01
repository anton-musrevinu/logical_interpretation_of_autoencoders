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
	elif opt.phase == 'graph':
		manager.make_graphs()
	elif opt.phase == 'encode':
		manager.convert_all_data(opt.task_type)
	elif opt.phase == 'create_impossible':
		manager.create_impossible_test_set(opt.task_type)
	elif opt.phase == 'decode':
		manager.decode_specific_file(opt.file_to_decode)
	elif opt.phase == 'examples':
		manager.make_class_examples()

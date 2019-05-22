import os,sys
sys.path.append('..')
from src import learn_psdd_wrapper as learn_psdd_wrapper
from src.lowlevel.util.psdd_interface import write_fl_batch_to_file,read_info_file
import numpy as np
import shutil

LOWLEVEL_CMD = '../src/lowlevel/main.py'
WMISDD_CMD = '../src/wmisdd/wmisdd.py'

def learn_encoder(testing = False):
	if testing:
		os.system('python {} --phase train --experiment_name {} --dataset {} --num_batches 50 --num_epochs 2 --batch_size 50 --feature_layer_size 28'.format(LOWLEVEL_CMD, experiment_name,dataset))
	else:
		os.system('python {} --phase train --experiment_name {} --gpu_ids 0,1 --feature_layer_size 1 --categorical_dim 10 --num_epochs 200'.format(LOWLEVEL_CMD, experiment_name))

def encode_data(experiment_name, encoded_data_dir, testing = False, compress_fly = True, task_type = 'classification'):
	if testing:
		os.system('python {} --phase encode --experiment_name {} --encoded_data_dir {} --limit_conversion 300 --compress_fly {} --task_type {}'.format(LOWLEVEL_CMD, experiment_name,encoded_data_dir, compress_fly, task_type))
	else:
		os.system('python {} --phase encode --experiment_name {} --encoded_data_dir {} --compress_fly {} --task_type {}'.format(LOWLEVEL_CMD, experiment_name,encoded_data_dir, compress_fly,task_type))

def decode_data(experiment_name, file_to_decode):
	print(experiment_name)
	cmd = 'python {} --phase decode --experiment_name {} --file_to_decode {}'.format(LOWLEVEL_CMD, experiment_name, file_to_decode)
	print('executing: {}'.format(cmd))
	os.system(cmd)


def do_psdd_training(experiment_dir_path,cluster_name, compress_fly = True, small_data_set = False, \
					do_encode_data = True, num_compent_learners = 1, vtree_method = 'miBlossom', task_type = 'classification'):
	os.system('pwd')

	encoded_name = 'encoded_data' if compress_fly else 'encoded_data_uncompressed_y' 
	if task_type != 'classification':
		encoded_name += '_{}'.format(task_type)

	encoded_data_dir = os.path.join(experiment_dir_path,encoded_name)

	if do_encode_data:
		encode_data(experiment_dir_path.split('/')[-1], encoded_data_dir, testing = small_data_set, compress_fly = compress_fly, task_type = task_type)

	for root, dir_names, file_names in os.walk(encoded_data_dir):
		for i in file_names:
			if i.endswith('train.data'):
				train_data_path = os.path.join(root, i)
			elif i.endswith('valid.data'):
				valid_data_path = os.path.join(root, i)
			elif i.endswith('test.data'):
				test_data_file = os.path.join(root, i)

	y_constraints = None
	if not compress_fly:
		y_constraints = os.path.join(experiment_dir_path, './y_constraints.cnf')

	psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_search_{}/'.format(cluster_id))	

	vtree_path, psdds, weights = learn_psdd_wrapper.learn_psdd(psdd_out_dir, train_data_path, valid_data_path = valid_data_path,\
				replace_existing = True, vtree_method = vtree_method, num_compent_learners = num_compent_learners, constraints_cnf_file = y_constraints)

def do_classification_evaluation(experiment_dir_path, cluster_id, test = False, task_type = 'classification'):
	print('experiment_dir_path',experiment_dir_path)

	if 'ex_7_mnist_16_4' in experiment_dir_path and 'james10' == cluster_id:
		encoded_data_dir = os.path.join(experiment_dir_path,'encoded_data_uncompressed_y')
	elif task_type != 'classification':
		encoded_data_dir = os.path.join(experiment_dir_path, 'encoded_data_' + task_type)
	else:
		encoded_data_dir = os.path.join(experiment_dir_path,'encoded_data')

	for root, dir_names, file_names in os.walk(encoded_data_dir):
		for i in file_names:
			if i.endswith('train.data'):
				train_data_path = os.path.join(root, i)
			elif i.endswith('valid.data'):
				valid_data_path = os.path.join(root, i)
			elif i.endswith('test.data'):
				test_data_file = os.path.join(root, i)

	if 'ex_5' in experiment_dir_path or 'ex_6' in experiment_dir_path:
		psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_model_{}/'.format(cluster_id))
	else:
		psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_search_{}/'.format(cluster_id))

	for i in range(5):
		try:
			at_iteration = 'best-{}'.format(i)
			print('trying at: {}'.format(at_iteration))
			learn_psdd_wrapper.measure_classification_accuracy_on_file(psdd_out_dir, test_data_file, train_data_path, valid_data_path = valid_data_path, \
									test = test, psdd_init_data_per = 0.1 if not test else 0.01, at_iteration = at_iteration)
			break
		except Exception as e:
			print('caught exception: {}'.format(e))
			continue

def do_generative_query_on_test(experiment_dir_path, cluster_id, test = False, task_type = 'classification', \
	fl_to_query = ['flx'], type_of_query = 'dis', y_condition = None):
	print('experiment_dir_path',experiment_dir_path)


	if 'ex_7_mnist_16_4' in experiment_dir_path and 'james10' == cluster_id:
		encoded_data_dir = os.path.join(experiment_dir_path,'encoded_data_uncompressed_y')
	elif task_type != 'classification':
		encoded_data_dir = os.path.join(experiment_dir_path, 'encoded_data_' + task_type)
	else:
		encoded_data_dir = os.path.join(experiment_dir_path,'encoded_data')

	for root, dir_names, file_names in os.walk(encoded_data_dir):
		for i in file_names:
			if i.endswith('train.data'):
				train_data_path = os.path.join(root, i)
			elif i.endswith('valid.data'):
				valid_data_path = os.path.join(root, i)
			elif i.endswith('test.data'):
				query_data_file = os.path.join(root, i)

	if 'ex_5' in experiment_dir_path or 'ex_6' in experiment_dir_path:
		psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_model_{}/'.format(cluster_id))
	else:
		psdd_out_dir = os.path.abspath(os.path.join(experiment_dir_path,'./psdd_search_{}/'.format(cluster_id)))

	# generated_data_file = learn_psdd_wrapper.generative_query_for_file(psdd_out_dir, query_data_file, train_data_path, valid_data_path = valid_data_path, \
	# 	test = test, psdd_init_data_per = 0.1, type_of_query = type_of_query, fl_to_query = fl_to_query, y_condition = y_condition)

	# decode_data(experiment_dir_path.split('/')[-1], generated_data_file)

	_decode_class_samples(psdd_out_dir, cluster_id)

def do_generative_query_for_labels(experiment_dir_path, cluster_id, test = False, type_of_query = 'dis'):
	print('experiment_dir_path',experiment_dir_path)
	encoded_data_dir = os.path.join(experiment_dir_path,'encoded_data')
	for root, dir_names, file_names in os.walk(encoded_data_dir):
		for i in file_names:
			if i.endswith('train.data'):
				train_data_path = os.path.join(root, i)
			elif i.endswith('valid.data'):
				valid_data_path = os.path.join(root, i)
			elif i.endswith('test.data'):
				test_data_file = os.path.join(root, i)

	if 'ex_5' in experiment_dir_path or 'ex_6' in experiment_dir_path:
		psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_model_{}/'.format(cluster_id))
	else:
		psdd_out_dir = os.path.abspath(os.path.join(experiment_dir_path,'./psdd_search_{}/'.format(cluster_id)))

	evaluationDir = os.path.abspath(os.path.join(psdd_out_dir, './evaluation/'))
	if not learn_psdd_wrapper._check_if_dir_exists(evaluationDir, raiseException = False):
		os.mkdir(evaluationDir)

	fl_info = read_info_file(test_data_file)

	for i in range(10):
		file_name = os.path.abspath(os.path.join(evaluationDir, 'query_label_{}.data'.format(i)))
		y_vec = np.zeros((100,10))
		y_vec[:,i] += 1
		write_fl_batch_to_file(file_name, np.zeros((100, fl_info['flx'].nb_vars, fl_info['flx'].var_cat_dim)), y_vec, 0)

		learn_psdd_wrapper.generative_query_for_file(psdd_out_dir, file_name, train_data_path, valid_data_path = valid_data_path, \
						test = False, psdd_init_data_per = 0.1 if not test else 0.01, type_of_query = type_of_query)
	_decode_class_samples(psdd_out_dir, cluster_id)

# ============================================================================================================================
# ============================================================================================================================
# ============================================================================================================================
def _decode_all_possible(experiment_dir_path, cluster_id):
	psdd_out_dir = os.path.abspath(os.path.join(experiment_dir_path,'./psdd_search_{}/'.format(cluster_id)))
	_decode_class_samples(psdd_out_dir, cluster_id)

def _decode_class_samples(psdd_out_dir, cluster_id):
	evaluationDir = os.path.abspath(os.path.join(psdd_out_dir, './evaluation/'))
	if not os.path.exists(evaluationDir):
		raise Exception('no samples could be found')

	files_to_decode = []
	for root, dir_names, file_names in os.walk(evaluationDir):
		for i in file_names:
			if 'generated' in i and i.endswith('.data'):
				files_to_decode.append(os.path.join(root, i))
				# info_file = os.path.join(root, i + '.info')
				# if not os.path.exists(info_file):
				# 	original_info_file = os.path.join(root, i.split('-')[0] + '.data.info')
				# 	shutil.copyfile(original_info_file, info_file)

	for file in files_to_decode:
		decode_data(psdd_out_dir.split('/')[-2], file)
		# return

def evaluate_all_missing():
	base_dir = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/'))

	toclassify = []
	for root, dir_names, file_names in os.walk(base_dir):
		for exp in dir_names:
			for experiment_dir_path, dir_names_2, file_names_2 in os.walk(os.path.join(root, exp)):
				for psdd_search_dir in dir_names_2:
					if 'psdd_search' in psdd_search_dir:
						cluster_id = str(psdd_search_dir).split('psdd_search_')[1].replace('/','')
						exp_cluster_dir = os.path.abspath(os.path.join(experiment_dir_path, psdd_search_dir))
						evaluationDir = os.path.abspath(os.path.join(exp_cluster_dir, './evaluation'))
						if not os.path.exists(evaluationDir):
							toclassify.append((experiment_dir_path, cluster_id))
							# print('added because evaldir does not exist', evaluationDir)
						else:
							for root_3, dir_names_3, file_names_3 in os.walk(evaluationDir):
								if not any(['classification' in file_name for file_name in file_names_3]):
									toclassify.append((experiment_dir_path, cluster_id))
									# print('added because file does not exist', (experiment_dir_path, cluster_id))

	for i in toclassify:
		print('evaluating ', i)
		do_classification_evaluation(i[0],i[1])

def do_everything(experiment_dir_path,cluster_id, \
		vtree_method = 'miBlossom', num_compent_learners = 10, compress_fly = True, type_of_query = 'bin', task_type = 'classification',\
		do_encode_data = True):
	
	#encode the data for and learn the (ensemby psdd)
	do_psdd_training(experiment_dir_path, cluster_id, do_encode_data = do_encode_data, \
	 	compress_fly = compress_fly, num_compent_learners = num_compent_learners , vtree_method = vtree_method, task_type = task_type)

	#record classification acc on held out test set
	do_classification_evaluation(experiment_dir_path, cluster_id, task_type = task_type)

	#Generate class samples and decode them to png
	do_generative_query_on_test(experiment_dir_path, cluster_id, task_type = task_type, type_of_query = type_of_query,test = True, fl_to_query = ['fla'],y_condition = [0])
	do_generative_query_on_test(experiment_dir_path, cluster_id, task_type = task_type, type_of_query = type_of_query,test = True, fl_to_query = ['fla'],y_condition = [1])


if __name__ == '__main__':

	# evaluate_all_missing()

	experiment_name = 'ex_7_mnist_32_2'
	cluster_id = 'james03_plus'
	task_type = 'plus'
	# compress_fly = True
	experiment_dir_path = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/{}'.format(experiment_name)))

	# encoded_name = 'encoded_data' if compress_fly else 'encoded_data_uncompressed_y' 
	# if task_type != 'classification':
	# 	encoded_name += '_{}'.format(task_type)

	# encoded_data_dir = os.path.join(experiment_dir_path,encoded_name)

	# encode_data(experiment_dir_path.split('/')[-1], encoded_data_dir, testing = True, compress_fly = True, task_type = task_type)

	do_everything(experiment_dir_path, cluster_id, task_type = task_type, vtree_method = 'miGreedyBU', do_encode_data = True)
	


	# experiment_name = 'ex_7_mnist_24_4'
	# cluster_id = 'staff_compute'
	# experiment_dir_path = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/{}'.format(experiment_name)))
	# # do_psdd_training(experiment_dir_path, cluster_id, small_data_set = False, do_encode_data = False, \
	# # 	compress_fly = True, num_compent_learners = 10 , vtree_method = 'miBlossom')
	
	# #test classification acc on held out test set
	# do_classification_evaluation(experiment_dir_path, cluster_id, test = True, task_type = task_type)

	# #Generate class samples and decode them to png
	# do_generative_query_for_labels(experiment_dir_path, cluster_id, type_of_query = 'bin', task_type = task_type)
	# do_generative_query_on_test(experiment_dir_path, cluster_id, type_of_query = 'bin', task_type = task_type, test = True, fl_to_query = ['fla'], y_condition = [0])
	# do_generative_query_on_test(experiment_dir_path, cluster_id, type_of_query = 'bin', task_type = task_type, test = True, fl_to_query = ['fla'], y_condition = [1])
	# do_generative_query_on_test(experiment_dir_path, cluster_id, type_of_query = 'bin', task_type = task_type, test = True, fl_to_query = ['flb'])
	# _decode_class_samples(experiment_dir_path, )



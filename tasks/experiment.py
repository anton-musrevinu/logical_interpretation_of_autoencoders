import os,sys
sys.path.append('..')
from src import learn_psdd_wrapper as learn_psdd_wrapper
from src.lowlevel.util.psdd_interface import read_info_file
from src.lowlevel.util.psdd_interface import write_fl_batch_to_file_new as write_fl_batch_to_file
import numpy as np
import shutil

LOWLEVEL_CMD = '../src/lowlevel/main.py'
WMISDD_CMD = '../src/wmisdd/wmisdd.py'

class Experiment(object):
	def __init__(self, experiment_parent_name, cluster_id, task_type, compress_fly = None):
		self.experiment_parent_name = experiment_parent_name
		self.cluster_id = cluster_id
		self.task_type = task_type
		self.compress_fly = compress_fly

		self.experiment_dir_path = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/{}'.format(self.experiment_parent_name)))
		self.type_of_data = 'emnist' if '_emnist_' in self.experiment_parent_name else 'mnist'

		self.psdd_out_dir = self._get_psdd_out_dir(self.experiment_dir_path, self.cluster_id, self.task_type)
		self.encoded_data_dir = self._get_encoded_data_dir()

		self.fl_info = None
		self.set_fl_info_after_enoding()

		self.evaluation_dir_path = os.path.abspath(os.path.join(self.psdd_out_dir, './evaluation/'))

	def set_fl_info_after_enoding(self):
		fl_data_file = os.path.join(self.psdd_out_dir, './fl_data.info')
		if os.path.exists(fl_data_file):
			fl_info = read_info_file(fl_data_file.replace('.info', ''))
			self.fl_info = fl_info
			if 'fly' in self.fl_info:
				self.compress_fly = self.fl_info['fly'].bin_encoded
		else:
			print('fl_info file could not be found at this point')

	def _get_encoded_data_dir(self):
		encoded_data_dir = None
		if os.path.exists(self.psdd_out_dir):
			fl_data_file = os.path.join(self.psdd_out_dir, './fl_data.info')
			if os.path.exists(fl_data_file):
				identifier = 'encoded_data_dir:'
				with open(fl_data_file, 'r') as f:
					for line in f:
						if line.startswith(identifier):
							relative_path = line.split(identifier)[-1].strip()
							joined = os.path.join(self.psdd_out_dir, relative_path)
							encoded_data_dir = os.path.abspath(joined)
			else:
				if 'ex_7_mnist_16_4' in self.experiment_dir_path and 'james10' == self.cluster_id or\
					'ex_7_mnist_32_2' in self.experiment_dir_path and 'student_compute' == self.cluster_id:
					encoded_data_dir = os.path.join(self.experiment_dir_path,'encoded_data_uncompressed_y')
				elif self.task_type != 'classification':
					task_type = self.task_type
					if self.task_type == 'succ':
						task_type = 'successor'
					encoded_data_dir = os.path.join(self.experiment_dir_path, 'encoded_data_' + task_type)
				else:
					encoded_data_dir = os.path.join(self.experiment_dir_path,'encoded_data')
		else:
			if self.compress_fly == None:
				raise Exception('trying to create a new experiment but compress_fly was not provided')
			experiment_identifier = ''
			if not self.compress_fly:
				experiment_identifier += '_uncompressed_y'
			if self.task_type != 'classification':
				experiment_identifier += '_' + self.task_type
			encoded_data_dir = os.path.join(self.experiment_dir_path, 'encoded_data' + experiment_identifier)

		return encoded_data_dir

	def _get_psdd_out_dir(self, experiment_dir_path, cluster_id, task_type):
		identifier = cluster_id + '_' + task_type if task_type != 'classification' else cluster_id

		if 'ex_5' in experiment_dir_path or 'ex_6' in experiment_dir_path:
			psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_model_{}'.format(identifier))
		else:
			psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_search_{}'.format(identifier))

		return os.path.abspath(psdd_out_dir)

	def get_data_files(self):
		print('Searching {} for data files'.format(self.encoded_data_dir))
		for root, dir_names, file_names in os.walk(self.encoded_data_dir):
			for i in file_names:
				if i.endswith('train.data'):
					train_data_path = os.path.join(root, i)
				elif i.endswith('valid.data'):
					valid_data_path = os.path.join(root, i)
				elif i.endswith('test.data'):
					test_data_path = os.path.join(root, i)

		return train_data_path, valid_data_path, test_data_path

# def _get_psdd_out_dir(experiment_dir_path, cluster_id, task_type):
# 	identifier = cluster_id + '_' + task_type if task_type != 'classification' else cluster_id

# 	if 'ex_5' in experiment_dir_path or 'ex_6' in experiment_dir_path:
# 		psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_model_{}'.format(identifier))
# 	else:
# 		psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_search_{}'.format(identifier))

# 	return os.path.abspath(psdd_out_dir)

# def _get_data_files(experiment_dir_path, cluster_id, task_type):
	# encoded_data_dir = None

	# #Read info file located at tte root of psdd_out_dir
	# psdd_out_dir = _get_psdd_out_dir(experiment_dir_path, cluster_id, task_type)
	# # print('psdd out dir: {}, {}, {}, {}'.format(psdd_out_dir, experiment_dir_path, cluster_id, task_type))
	# fl_data_file = os.path.join(psdd_out_dir, './fl_data.info')
	# if os.path.exists(fl_data_file):
	# 	identifier = 'encoded_data_dir:'
	# 	with open(fl_data_file, 'r') as f:
	# 		for line in f:
	# 			if line.startswith(identifier):
	# 				relative_path = line.split(identifier)[-1].strip()
	# 				joined = os.path.join(psdd_out_dir, relative_path)
	# 				encoded_data_dir = os.path.abspath(joined)

	# #Reconstruct the information in case the file does not exists (old experiemnts)
	# if encoded_data_dir == None:
	# 	if 'ex_7_mnist_16_4' in experiment_dir_path and 'james10' == cluster_id or\
	# 		'ex_7_mnist_32_2' in experiment_dir_path and 'student_compute' == cluster_id:
	# 		encoded_data_dir = os.path.join(experiment_dir_path,'encoded_data_uncompressed_y')
	# 	elif task_type != 'classification':
	# 		if task_type == 'succ':
	# 			task_type = 'successor'
	# 		encoded_data_dir = os.path.join(experiment_dir_path, 'encoded_data_' + task_type)
	# 	else:
	# 		encoded_data_dir = os.path.join(experiment_dir_path,'encoded_data')

	# print('Searching {} for data files'.format(encoded_data_dir))
	# for root, dir_names, file_names in os.walk(encoded_data_dir):
	# 	for i in file_names:
	# 		if i.endswith('train.data'):
	# 			train_data_path = os.path.join(root, i)
	# 		elif i.endswith('valid.data'):
	# 			valid_data_path = os.path.join(root, i)
	# 		elif i.endswith('test.data'):
	# 			test_data_path = os.path.join(root, i)

	# return train_data_path, valid_data_path, test_data_path

# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

def learn_encoder(experiment_name, dataset = 'mnist',testing = False, ):
	if testing:
		os.system('python {} --phase train --experiment_name {} --dataset {} --num_batches 50 --num_epochs 2 --batch_size 100 --feature_layer_size 28'.format(LOWLEVEL_CMD, experiment_name,dataset))
	else:
		os.system('python {} --phase train --experiment_name {} --dataset {} --gpu_ids 0,1 --feature_layer_size 32 --categorical_dim 2 --num_epochs 400 --batch_size 100'.format(LOWLEVEL_CMD, experiment_name,dataset))

def encode_data(exp, testing = False):
	if testing:
		os.system('python {} --phase encode --experiment_name {} --encoded_data_dir {} --limit_conversion 300 --compress_fly {} --task_type {}'.format(LOWLEVEL_CMD, exp.experiment_parent_name, exp.encoded_data_dir, exp.compress_fly, exp.task_type))
	else:
		os.system('python {} --phase encode --experiment_name {} --encoded_data_dir {} --compress_fly {} --task_type {}'.format(LOWLEVEL_CMD, exp.experiment_parent_name, exp.encoded_data_dir, exp.compress_fly, exp.task_type))

def graph_learning(experiment_parent_name):
	os.system('python {} --experiment_name {} --phase graph'.format(LOWLEVEL_CMD,experiment_parent_name))

def decode_data(exp, file_to_decode):
	cmd = 'python {} --phase decode --experiment_name {} --file_to_decode {}'.format(LOWLEVEL_CMD, exp.experiment_parent_name, file_to_decode)
	print('executing: {}'.format(cmd))
	os.system(cmd)

def do_psdd_training(exp, testing = False, do_encode_data = True, num_compent_learners = 1, vtree_method = 'miBlossom'):

	#Name of the folder for the encoded data (cluster independent)
	if do_encode_data:
		encode_data(exp, testing = testing)

	train_data_path, valid_data_path, test_data_file = exp.get_data_files()

	y_constraints = None
	#CONSTRAINTS ARE NOT WORKING FOR ENSEMBLY PSDD!!!!!!!!!!!!!!!!!!!!
	# if not compress_fly:
	# 	y_constraints = os.path.join(experiment_dir_path, './y_constraints.cnf')

	learn_psdd_wrapper.learn_psdd(exp.psdd_out_dir, train_data_path, valid_data_path = valid_data_path,\
				replace_existing = True, vtree_method = vtree_method, num_compent_learners = num_compent_learners, constraints_cnf_file = y_constraints)

	exp.set_fl_info_after_enoding()

# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

def do_classification_evaluation(exp, test = False):
	print('[CLASSIFICATION] - START ON: \t{}'.format(exp))

	try:
		train_data_path, valid_data_path, query_data_path = exp.get_data_files()
	except Exception as e:
		print(e)
		print('[CLASSIFICATION] - END DUE: \tNeccesary data files could not be loaded')
		return

	for i in range(10):
		try:
			at_iteration = 'best-{}'.format(i)
			print('trying at: {}'.format(at_iteration))
			learn_psdd_wrapper.measure_classification_accuracy_on_file(exp.psdd_out_dir, query_data_path, train_data_path, valid_data_path = valid_data_path, \
									test = test, psdd_init_data_per = 0.1 if not test else 0.01, at_iteration = at_iteration)
			break
		except Exception as e:
			print('caught exception: {}'.format(e))
			continue

def do_generative_query(exp, test = False, type_of_query = 'bin'):

	print('[DOINGIN GENERATIVE QUERY WITH ARGS: {}, --  {}'.format(exp, type_of_query))
	
	if exp.task_type == 'classification' or 'noisy' in exp.task_type:
		do_generative_query_for_labels(exp, type_of_query = type_of_query)
	elif exp.task_type == 'succ':
		do_generative_query_on_test(exp, type_of_query = type_of_query, test = True, fl_to_query = ['fla'])
		do_generative_query_on_test(exp, type_of_query = type_of_query, test = True, fl_to_query = ['flb'])
	else:
		#Generate class samples and decode them to png
		do_generative_query_on_test(exp, type_of_query = type_of_query, test = True, fl_to_query = ['fla'], y_condition = [0])
		do_generative_query_on_test(exp, type_of_query = type_of_query, test = True, fl_to_query = ['fla'], y_condition = [1])

	do_decode_class_samples(exp)

def do_generative_query_on_test(exp, test = False, \
	fl_to_query = ['flx'], type_of_query = 'dis', y_condition = None):

	print('[SAMPLING] - START ON: \t{} -- {}'.format(exp, type_of_query))

	try:
		train_data_path, valid_data_path, query_data_path = exp.get_data_files()
	except Exception as e:
		print('[SAMPLING] - END DUE: \tNeccesary data files could not be found')
		return

	for i in range(5):
		try:
			at_iteration = 'best-{}'.format(i)
			print('trying at: {}'.format(at_iteration))
			learn_psdd_wrapper.generative_query_for_file(exp.psdd_out_dir, query_data_path, train_data_path, valid_data_path = valid_data_path, \
				test = test, psdd_init_data_per = 0.1, type_of_query = type_of_query, fl_to_query = fl_to_query, y_condition = y_condition, at_iteration = at_iteration)
			break
		except Exception as e:
			print('caught exception: {}'.format(e))
			continue

def do_generative_query_for_labels(exp, test = False, type_of_query = 'bin'):

	try:
		train_data_path, valid_data_path, query_data_path = exp.get_data_files()
	except Exception as e:
		print(e)
		print('[GENERATIVE] - END DUE: \tNeccesary data files could not be loaded')
		return

	evaluation_dir_path = exp.evaluation_dir_path
	if not learn_psdd_wrapper._check_if_dir_exists(evaluation_dir_path, raiseException = False):
		os.mkdir(evaluation_dir_path)

	fl_info = exp.fl_info
	fl_to_query = ['flx'] if task_type == 'classification' else ['fla']

	for fl_y_value in range(fl_info['fly'].var_cat_dim):
		file_name = os.path.abspath(os.path.join(evaluation_dir_path, 'query_label_{}.data'.format(fl_y_value)))
		fl_data = {}
		for key in fl_info.keys():
			if key in fl_to_query:
				data = np.zeros((100, fl_info[key].nb_vars, fl_info[key].var_cat_dim))
			elif key == 'fly':
				data = np.zeros((100, fl_info[key].var_cat_dim))
				data[:,fl_y_value] += 1
			fl_data[key] = data
		write_fl_batch_to_file(file_name, fl_data, fl_info, 0)

		for best_i in range(5):
			try:
				at_iteration = 'best-{}'.format(best_i)
				print('trying at: {}'.format(at_iteration))
				learn_psdd_wrapper.generative_query_for_file(exp.psdd_out_dir, file_name, train_data_path, valid_data_path = valid_data_path, \
					test = test, psdd_init_data_per = 0.1 if not test else 0.01, type_of_query = exp.type_of_query, fl_to_query = fl_to_query, at_iteration = at_iteration)
				break
			except Exception as e:
				print('caught exception: {}'.format(e))
				continue

def do_decode_class_samples(exp):

	if not os.path.exists(exp.evaluation_dir_path):
		raise Exception('no samples could be found')

	files_to_decode = []
	to_remove = []
	for root, dir_names, file_names in os.walk(exp.evaluation_dir_path):
		for i in file_names:
			if 'generated' in i and i.endswith('.data') and os.path.getsize(os.path.join(root, i)) != 0:
				if not any([f.startswith(i.replace('.data','')) and f.endswith('.png') for f in file_names]):
					files_to_decode.append(os.path.join(root, i))
			if 'generated' in i and i.endswith('.data') and os.path.getsize(os.path.join(root, i)) == 0:
				to_remove.append(os.path.join(root, i))
				# info_file = os.path.join(root, i + '.info')
				# if not os.path.exists(info_file):
				# 	original_info_file = os.path.join(root, i.split('-')[0] + '.data.info')
				# 	shutil.copyfile(original_info_file, info_file)

	# for file in to_remove:
	# 	print(file)

	for file in files_to_decode:
		decode_data(exp, file)
# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================
def decode_all_possible():
	base_dir = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/'))

	toclassify = []
	for root, dir_names, file_names in os.walk(base_dir):
		for experiment_parent_name in dir_names:
			for experiment_dir_path, dir_names_2, file_names_2 in os.walk(os.path.join(root, experiment_parent_name)):
				for psdd_search_dir in dir_names_2:
					if 'psdd_search' in psdd_search_dir:
						identifier = str(psdd_search_dir).split('psdd_search_')[1].replace('/','')
						if len(identifier.split('_')) == 3 or (len(identifier.split('_')) == 2 and 'james' in identifier):
							cluster_id = '_'.join(identifier.split('_')[:-1])
							task_type = identifier.split('_')[-1]
						else:
							cluster_id = identifier
							task_type = 'classification'
						exp_cluster_dir = os.path.abspath(os.path.join(experiment_dir_path, psdd_search_dir))
						progressfile = os.path.join(exp_cluster_dir, './learnpsdd_tmp_dir/progress.txt')

						if not os.path.exists(progressfile) or os.path.getsize(progressfile) == 0:
							continue

						evaluationDir = os.path.abspath(os.path.join(exp_cluster_dir, './evaluation'))
						if not os.path.exists(evaluationDir):
							continue
							# print('added because evaldir does not exist', evaluationDir)
						exp = Experiment(experiment_parent_name, cluster_id, task_type)
						toclassify.append(exp)

	for exp in toclassify:
		do_decode_class_samples(exp)
		# break

def evaluate_all_missing():
	base_dir = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/'))

	toclassify = []
	for root, dir_names, file_names in os.walk(base_dir):
		for experiment_parent_name in dir_names:
			for experiment_dir_path, dir_names_2, file_names_2 in os.walk(os.path.join(root, experiment_parent_name)):
				for psdd_search_dir in dir_names_2:
					if 'psdd_search' in psdd_search_dir:
						identifier = str(psdd_search_dir).split('psdd_search_')[1].replace('/','')
						if len(identifier.split('_')) == 3 or (len(identifier.split('_')) == 2 and 'james' in identifier):
							cluster_id = '_'.join(identifier.split('_')[:-1])
							task_type = identifier.split('_')[-1]
						else:
							cluster_id = identifier
							task_type = 'classification'

						exp = Experiment(experiment_parent_name, cluster_id, task_type)

						progressfile = os.path.join(exp.psdd_out_dir, './learnpsdd_tmp_dir/progress.txt')
						if not os.path.exists(progressfile) or os.path.getsize(progressfile) == 0:
							continue

						if not os.path.exists(exp.evaluation_dir_path):
							toclassify.append(exp)
							# print('added because evaldir does not exist', evaluationDir)
						else:
							for root_3, dir_names_3, file_names_3 in os.walk(exp.evaluation_dir_path):
								if not any(['classification' in file_name for file_name in file_names_3]):
									toclassify.append(exp)
								else:
									allempty = True
									for file_4 in file_names_3:
										if 'classification' in file_4:
											allempty = allempty and os.path.getsize(os.path.join(root_3, file_4)) == 0
									if allempty:
										toclassify.append(exp)
									# print('added because file does not exist', (experiment_dir_path, cluster_id))

	for exp in toclassify:
		if exp.task_type == 'succ':
			continue
		do_classification_evaluation(exp)
		# break

def sample_all_missing():
	base_dir = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/'))

	toclassify = []
	for root, dir_names, file_names in os.walk(base_dir):
		for experiment_parent_name in dir_names:
			for experiment_dir_path, dir_names_2, file_names_2 in os.walk(os.path.join(root, experiment_parent_name)):
				for psdd_search_dir in dir_names_2:
					if 'psdd_search' in psdd_search_dir:
						identifier = str(psdd_search_dir).split('psdd_search_')[1].replace('/','')
						if len(identifier.split('_')) == 3 or (len(identifier.split('_')) == 2 and 'james' in identifier):
							cluster_id = '_'.join(identifier.split('_')[:-1])
							task_type = identifier.split('_')[-1]
						else:
							cluster_id = identifier
							task_type = 'classification'

						exp = Experiment(experiment_parent_name, cluster_id, task_type)

						progressfile = os.path.join(exp.psdd_out_dir, './learnpsdd_tmp_dir/progress.txt')
						if not os.path.exists(progressfile) or os.path.getsize(progressfile) == 0:
							continue

						if not os.path.exists(exp.evaluation_dir_path):
							toclassify.append((exp, 'bin'))
							toclassify.append((exp, 'dis'))
							# print('added because evaldir does not exist', evaluationDir)
						else:
							for root_3, dir_names_3, file_names_3 in os.walk(exp.evaluation_dir_path):
								if not any(['generated' in file_name and file_name.endswith('.png') and 'binb' in file_name for file_name in file_names_3]):
									toclassify.append((exp, 'bin'))
								if not any(['generated' in file_name and file_name.endswith('.png') and 'disb' in file_name for file_name in file_names_3]):
									toclassify.append((exp, 'dis'))


									# print('added because file does not exist', (experiment_dir_path, cluster_id))

	print('sampling for experiments:')
	for (exp, type_of_query) in toclassify:
		if exp.task_type != 'classification' and exp.task_type != 'plus':
			do_generative_query(exp, type_of_query = type_of_query)
	for (exp, type_of_query) in toclassify:
		if exp.task_type == 'classification':
			do_generative_query(exp, type_of_query = type_of_query)

def make_learning_graphs_missing():
	base_dir = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/'))

	tograph = []
	for root, dir_names, file_names in os.walk(base_dir):
		for experiment_parent_name in dir_names:
			tograph.append(experiment_parent_name)

	for experiment_parent_name in tograph:
		graph_learning(experiment_parent_name)

# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

def do_everything(exp, vtree_method = 'miBlossom', num_compent_learners = 10, type_of_query = 'bin', do_encode_data = True, testing = False):
	
	#encode the data for and learn the (ensemby psdd)
	do_psdd_training(exp, do_encode_data = do_encode_data, num_compent_learners = num_compent_learners , vtree_method = vtree_method, testing = testing)

	#record classification acc on held out test set
	do_classification_evaluation(exp, testing = testing)

	#Generate class samples and decode them to png
	do_generative_query(exp, testing = testing, type_of_query = type_of_query)
	

if __name__ == '__main__':

	# decode_all_possible()
	# evaluate_all_missing()
	# sample_all_missing()

	experiment_parent_name = 'ex_6_emnist_32_2'
	cluster_id = 'james08'
	task_type = 'classification'
	compress_fly = False
	exp = Experiment(experiment_parent_name, cluster_id, task_type, compress_fly = compress_fly)
	# do_everything(exp)
	do_classification_evaluation(exp)
	# experiment_dir_path = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/{}'.format(experiment_name)))

	# # do_everything(experiment_dir_path, cluster_id, task_type = task_type, vtree_method = 'miBlossom', do_encode_data = True, testing = False,  \
	# # 				compress_fly = compress_fly)

	# do_generative_query(experiment_dir_path, cluster_id, task_type, type_of_query = 'dis')
	# do_generative_query_on_test(experiment_dir_path, cluster_id, task_type = task_type, type_of_query = 'bin', test = True, fl_to_query = ['fla'], y_condition = [0])
	# do_generative_query_on_test(experiment_dir_path, cluster_id, task_type = task_type, type_of_query = 'bin', test = True, fl_to_query = ['fla'], y_condition = [1])


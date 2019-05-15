import os,sys
sys.path.append('..')
from src import learn_psdd_wrapper as learn_psdd_wrapper
from src.lowlevel.util.psdd_interface import write_fl_batch_to_file,read_info_file
import numpy as np

LOWLEVEL_CMD = '../src/lowlevel/main.py'
WMISDD_CMD = '../src/wmisdd/wmisdd.py'

def learn_encoder(testing = False):
	if testing:
		os.system('python {} --phase train --experiment_name {} --dataset {} --num_batches 50 --num_epochs 2 --batch_size 50 --feature_layer_size 28'.format(LOWLEVEL_CMD, experiment_name,dataset))
	else:
		os.system('python {} --phase train --experiment_name {} --gpu_ids 0,1 --feature_layer_size 1 --categorical_dim 10 --num_epochs 200'.format(LOWLEVEL_CMD, experiment_name))

def encode_data(experiment_name, testing = False, compress_fly = True):
	if testing:
		os.system('python {} --phase encode --experiment_name {} --limit_conversion 100 --compress_fly {}'.format(LOWLEVEL_CMD, experiment_name, compress_fly))
	else:
		os.system('python {} --phase encode --experiment_name {} --compress_fly {}'.format(LOWLEVEL_CMD, experiment_name, compress_fly))

def decode_data(experiment_name, file_to_decode):
	print(experiment_name)
	cmd = 'python {} --phase decode --experiment_name {} --file_to_decode {}'.format(LOWLEVEL_CMD, experiment_name, file_to_decode)
	print('executing: {}'.format(cmd))
	os.system(cmd)


def do_psdd_training(experiment_dir_path,cluster_name, compress_fly = True, small_data_set = False, \
					do_encode_data = True, num_compent_learners = 1, vtree_method = 'miBlossom'):
	os.system('pwd')

	encoded_data_dir = os.path.join(experiment_dir_path,'encoded_data')

	if do_encode_data:
		encode_data(experiment_dir_path.split('/')[-1], testing = small_data_set, compress_fly = compress_fly)

	for root, dir_names, file_names in os.walk(encoded_data_dir):
		for i in file_names:
			if i.endswith('train.data'):
				train_data_path = os.path.join(root, i)
			elif i.endswith('valid.data'):
				valid_data_path = os.path.join(root, i)
			elif i.endswith('test.data'):
				test_data_file = os.path.join(root, i)

	psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_search_{}/'.format(cluster_id))
	vtree_path, psdds, weights = learn_psdd_wrapper.learn_psdd(psdd_out_dir, train_data_path, valid_data_path = valid_data_path,\
				replace_existing = True, vtree_method = vtree_method, num_compent_learners = num_compent_learners)

def do_classification_evaluation(experiment_dir_path, cluster_id, test = False):
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
		psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_search_{}/'.format(cluster_id))

	learn_psdd_wrapper.measure_classifcation_accuracy_on_file(psdd_out_dir, test_data_file, train_data_path, valid_data_path = valid_data_path, test = test, psdd_init_data_per = 0.1 if not test else 0.01)

def do_generative_query_on_test(experiment_dir_path, cluster_id, test = False):
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
		psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_search_{}/'.format(cluster_id))

	learn_psdd_wrapper.generative_query_for_file(psdd_out_dir, test_data_file, train_data_path, valid_data_path = valid_data_path, test = test, psdd_init_data_per = 0.1 if not test else 0.01)

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
		psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_search_{}/'.format(cluster_id))

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
						test = False, psdd_init_data_per = 0.1 if not test else 0.01,type_of_query = type_of_query)

# ============================================================================================================================
# ============================================================================================================================
# ============================================================================================================================

def draw_class_samples(experiment_dir, cluster_id):

	# 18;1690.008417296;41879.36322377;2763;0.09;0.15;0.12;0.05;0.12;0.12;0.08;0.09;0.08;0.12;-44.754126036778328156475785

	vtree_file, list_of_psdds, list_of_weights, fly_catDim, flx_catDim, data_set_sample =\
		 get_experiment_info(experiment_dir, cluster_id)

	sampled_dir = os.path.join(experiment_dir, 'sampled_{}/'.format(cluster_id))
	if not os.path.exists(sampled_dir):
		os.mkdir(sampled_dir)

	# -v vtree
	# -p list of psdds
	# -a list of psdd weighs
	# -d data for initializing the psdd
	# -fly categorical dimention of the FLy --- the number of labels
	# -flx categorical dimention of the FLx
	# -o output file
	cmd = 'java -jar ' + LEARNPSDD_CMD + ' query -m generate -v {} -p {} -a {} -d {} -x {} -y {} -o {} -g {}'.format(\
		vtree_file,list_of_psdds, list_of_weights, data_set_sample, flx_catDim, fly_catDim,  sampled_dir, \
		'data_bug' in experiment_dir)
	print('excuting: {}'.format(cmd))
	os.system(cmd)

def decode_class_samples(experiment_dir, cluster_id):
	sampled_dir = os.path.join(experiment_dir, 'sampled_{}/'.format(cluster_id))
	if not os.path.exists(sampled_dir):
		raise Exception('no samples could be found')

	files_to_decode = {}
	for root, dir_names, file_names in os.walk(sampled_dir):
		for i in file_names:
			if 'samples_class_' in i and '.data' in i:
				class_id = int(i.split('_')[-1].split('.')[0])
				files_to_decode[class_id] = os.path.join(root, i)

	for class_id, file in files_to_decode.items():
		decode_data(experiment_name.split('/')[-1], file)
		# return

def classify_all_missing():

	experiment_name = 'ex_6_emnist_32_4'
	cluster_id = 'staff_compute'
	experiment_dir_path = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/{}'.format(experiment_name)))
	do_evaluation(experiment_dir_path, cluster_id)

	experiment_name = 'ex_6_emnist_32_4'
	cluster_id = 'james05'
	experiment_dir_path = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/{}'.format(experiment_name)))
	do_evaluation(experiment_dir_path, cluster_id)
	
	experiment_name = 'ex_5_mnist_32_4_data_bug'
	cluster_id = 'james03'
	experiment_dir_path = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/{}'.format(experiment_name)))
	do_evaluation(experiment_dir_path, cluster_id)	

	experiment_name = 'ex_5_mnist_64_4_data_bug'
	cluster_id = 'james02'
	experiment_dir_path = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/{}'.format(experiment_name)))
	do_evaluation(experiment_dir_path, cluster_id)
	


if __name__ == '__main__':
	experiment_name = 'ex_7_mnist_16_4'
	cluster_id = 'james01'
	experiment_dir_path = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/{}'.format(experiment_name)))
	do_psdd_training(experiment_dir_path, cluster_id, small_data_set = False, do_encode_data = False, \
		compress_fly = True, num_compent_learners = 10 , vtree_method = 'pairwiseWeights')
	
	# do_generative_query_for_labels(experiment_dir_path, cluster_id, test = True)
	# do_classification_evaluation(experiment_dir_path, cluster_id, test = True)

	# experiment_name = 'ex_5_mnist_32_4_data_bug'
	# cluster_id = 'student_compute'
	# experiment_dir_path = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/{}'.format(experiment_name)))
	# do_evaluation(experiment_dir_path, cluster_id, test = True)

	# measure_classifcation_acc(experiment_dir, cluster_id, test = False)
	# draw_class_samples(experiment_dir, cluster_id)
	# decode_class_samples(experiment_dir, cluster_id)




import os,sys
sys.path.append('..')
from src import learn_psdd_wrapper as learn_psdd_wrapper

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


def do_psdd_training(experiment_dir_path,cluster_name, compress_fly = True, small_data_set = False, do_encode_data = True):
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
				replace_existing = True, vtree_method = 'miBlossom')

def do_evaluation(experiment_dir_path, cluster_id, test = False):
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

	learn_psdd_wrapper.measure_classifcation_accuracy_on_file(psdd_out_dir, test_data_file, train_data_path, valid_data_path = valid_data_path, test = test, psdd_init_data_per = 0.1)

# ============================================================================================================================
# ============================================================================================================================
# ============================================================================================================================


def get_experiment_info(experiment_dir, cluster_id, test = False):
	if cluster_id == '':
		psdd_dir = os.path.join(experiment_dir, 'ensembly_psdd_model/')
		vtree_file = os.path.join(experiment_dir, 'symbolic_stuff/model_learned.vtree')
	else:
		psdd_dir = os.path.join(experiment_dir, 'psdd_model_{}/'.format(cluster_id))
		vtree_file = os.path.join(experiment_dir, 'symbolic_stuff_{}/model_learned.vtree'.format(cluster_id))

	for root, dir_names, file_names in os.walk(os.path.join(experiment_dir,'encoded_data')):
		for i in file_names:
			if 'train.data' in i and not 'sample' in i:
				train_data_file = os.path.join(root, i)

	fly_catDim = 10 if int(experiment_dir.split('/')[-1].split('_')[1]) <= 5 else 47
	flx_catDim = int(experiment_dir.split('/')[-1].split('_')[4])

	num_learners = -1
	latestIt = -1 
	weights = {}
	with open(os.path.join(psdd_dir, 'progress.txt'),'r') as prg:
		for line_idx, line in enumerate(prg):
			splitted = line.split(';')
			if len(splitted) < 5 or line_idx == 0:
				continue
			num_learners = len(splitted) - 5
			latestIt = int(splitted[0].strip())
			for idx, i in enumerate(range(4,num_learners + 4)):
				weights[idx] = float(splitted[i].strip())
	if latestIt == -1 or num_learners == -1:
		print('no iteration results found')
		return

	print('latest iteraion found: {}'.format(latestIt))
	print('corresponding weights: {}'.format(weights))

	list_of_psdds = ''
	list_of_weights = ''
	models = os.path.join(psdd_dir, 'models/')
	for i in range(num_learners):
		list_of_psdds = list_of_psdds + os.path.join(models, 'it_{}_l_{}.psdd'.format(latestIt,i)) + ','
		list_of_weights = list_of_weights + str(weights[i]) + ','
	list_of_psdds = list_of_psdds[:-1]
	list_of_weights = list_of_weights[:-1]

	data_set_sample = train_data_file + '.sample'
	if os.path.exists(data_set_sample):
		os.remove(data_set_sample)

	with open(data_set_sample, 'w') as f_to:
		with open(train_data_file, 'r') as f_from:
			for idx, line in enumerate(f_from):
				if idx < 100:
					f_to.write(line)
				if fly_catDim == 10:
					a = line.split(',')[-5]
					b = line.split(',')[-6]
					if a == '1' or b == '1':
						raise Exception('looks like we messed up') 

	return vtree_file, list_of_psdds, list_of_weights,fly_catDim, flx_catDim, data_set_sample

def measure_classifcation_acc(experiment_dir, cluster_id, test = False):

	# 18;1690.008417296;41879.36322377;2763;0.09;0.15;0.12;0.05;0.12;0.12;0.08;0.09;0.08;0.12;-44.754126036778328156475785

	vtree_file, list_of_psdds, list_of_weights, fly_catDim, flx_catDim, data_set_sample =\
			get_experiment_info(experiment_dir, cluster_id)

	evaluationDir = os.path.join(experiment_dir, 'evaluation_{}/'.format(cluster_id))
	if not os.path.exists(evaluationDir):
		os.mkdir(evaluationDir)
	outputFile = os.path.join(evaluationDir, 'classification.txt')

	if test:
		query = data_set_sample
	else:
		query = data_set_sample.replace('train.data.sample', 'test.data')
	# -v vtree
	# -p list of psdds
	# -a list of psdd weighs
	# -d data for initializing the psdd
	# -fly categorical dimention of the FLy --- the number of labels
	# -flx categorical dimention of the FLx
	# -o output file
	cmd = 'java -jar ' + LEARNPSDD_CMD + ' query -m classify -v {} -p {} -a {} -d {} -x {} -y {} -q {} -o {} -g {}'.format(\
		vtree_file,list_of_psdds, list_of_weights, data_set_sample, flx_catDim, fly_catDim, query,  outputFile,\
		'data_bug' in experiment_dir)
	print('excuting: {}'.format(cmd))
	os.system(cmd)

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
	# experiment_name = 'ex_5_mnist_64_4'
	# cluster_id = 'staff_compute'
	# experiment_dir_path = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/{}'.format(experiment_name)))
	# # do_psdd_training(experiment_dir_path, cluster_id, small_data_set = True, do_encode_data = False, compress_fly = True)
	# do_evaluation(experiment_dir_path, cluster_id)

	classify_all_missing()

	# experiment_name = 'ex_5_mnist_32_4_data_bug'
	# cluster_id = 'student_compute'
	# experiment_dir_path = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/{}'.format(experiment_name)))
	# do_evaluation(experiment_dir_path, cluster_id, test = True)

	# measure_classifcation_acc(experiment_dir, cluster_id, test = False)
	# draw_class_samples(experiment_dir, cluster_id)
	# decode_class_samples(experiment_dir, cluster_id)




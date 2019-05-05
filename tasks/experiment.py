import os

LOWLEVEL_CMD = '../src/lowlevel/main.py'
LEARNPSDD_CMD = '../src/Scala-LearnPsdd/target/scala-2.11/psdd.jar'
LEARNPSDD2_CMD = '../src/learnPSDD/target/scala-2.11/psdd.jar'
WMISDD_CMD = '../src/wmisdd/wmisdd.py'
SDD_CMD_DIR = '../src/wmisdd/bin/'

def learn_encoder(testing = False):
	if testing:
		os.system('python {} --phase train --experiment_name {} --dataset {} --num_batches 50 --num_epochs 2 --batch_size 50 --feature_layer_size 28'.format(LOWLEVEL_CMD, experiment_name,dataset))
	else:
		os.system('python {} --phase train --experiment_name {} --gpu_ids 0,1 --feature_layer_size 1 --categorical_dim 10 --num_epochs 200'.format(LOWLEVEL_CMD, experiment_name))

def encode_data(experiment_name, testing = False):
	if testing:
		os.system('python {} --phase encode --experiment_name {} --limit_conversion 100'.format(LOWLEVEL_CMD, experiment_name))
	else:
		os.system('python {} --phase encode --experiment_name {}'.format(LOWLEVEL_CMD, experiment_name))

def decode_data(experiment_name, file_to_decode):
	print(experiment_name)
	cmd = 'python {} --phase decode --experiment_name {} --file_to_decode {}'.format(LOWLEVEL_CMD, experiment_name, file_to_decode)
	print('executing: {}'.format(cmd))
	os.system(cmd)

def learn_vtree(train_data_file,vtree_file):
	# training_data = 
	# out_file = 

	cmd = 'java -jar {} learnVtree -d {} -o {}'.format(LEARNPSDD_CMD,train_data_file, vtree_file.replace('.vtree', ''))

	print('excuting: {}'.format(cmd))
	os.system(cmd)

	cmd_convert_to_pdf = 'dot -Tpdf {}.dot -o {}.pdf'.format(vtree_file,vtree_file)
	os.system(cmd_convert_to_pdf)

def compile_constraints_to_sdd(opt_file, sdd_file, vtree_file, total_num_variables, symbolic_dir,testing = False, precomputed_vtree = True):
	with open(opt_file, 'r') as f:
		for line in f:
			if 'feature_layer_size' in line:
				feature_layer_size = line.split(':')[1].split('[')[0].strip()
			elif 'categorical_dim' in line:
				categorical_dim = line.split(':')[1].split('[')[0].strip()

	print('feature_layer_size: {}, categorical_dim: {}, total_num_variables: {}'.format(feature_layer_size, categorical_dim, total_num_variables))

	cmd = 'python {} --mode onehot --onehot_numvars {} --onehot_fl_size {} --onehot_fl_categorical_dim {} --onehot_out_sdd {} --onehot_out_vtree {} --cnf_dir {} --precomputed_vtree {}'.format(\
				WMISDD_CMD, total_num_variables, feature_layer_size, categorical_dim, sdd_file, vtree_file, symbolic_dir, precomputed_vtree)
	if testing:
		cmd = 'python {} --mode onehot --onehot_numvars {} --onehot_fl_size {} --onehot_fl_categorical_dim {} --onehot_out_sdd {} --onehot_out_vtree {} --cnf_dir {} --precomputed_vtree False'.format(\
				WMISDD_CMD, 3*2 + 10, 3, 2, sdd_file, vtree_file, symbolic_dir)
	os.system(cmd)

	if precomputed_vtree == False:
		cmd_convert_to_pdf = 'dot -Tpdf {}.dot -o {}.pdf'.format(vtree_file, vtree_file)
		os.system(cmd_convert_to_pdf)
	cmd_convert_to_pdf = 'dot -Tpdf {}.dot -o {}.pdf'.format(sdd_file, sdd_file)
	os.system(cmd_convert_to_pdf)

def compile_sdd_to_psdd(train_data_file, valid_data_file, test_data_file, vtree_file, sdd_file, psdd_file):
	java_library_path = os.path.abspath(SDD_CMD_DIR)
	print('java_library_path: {}'.format(java_library_path)) # -Djava.library.path {} 
	cmd = 'java -jar {} sdd2psdd -d {} -b {} -t {} -v {} -s {} -m l-1 -o {}'.format(
		LEARNPSDD_CMD,train_data_file, valid_data_file, test_data_file, vtree_file, sdd_file, psdd_file)

	print('excuting: {}'.format(cmd))
	os.system(cmd)


def learn_ensembly_psdd_from_data(train_data_file,valid_data_file, test_data_file, vtree_file, psdd_file, psdd_out_dir, num_components = 10):
	cmd = 'java -jar {} learnEnsemblePsdd softEM -d {} -b {} -t {} -v {} -m l-1 -o {} -c {}'.format(\
		LEARNPSDD_CMD, train_data_file, valid_data_file, test_data_file, vtree_file,psdd_out_dir,num_components)

	print('excuting: {}'.format(cmd))
	os.system(cmd)

def learn_ensembly_psdd_2_from_data(dataDir, vtree_file, psdd_out_dir, num_components = 10):
	cmd = 'java -jar {} SoftEM {} {} {} {}'.format(\
		LEARNPSDD2_CMD, dataDir, vtree_file, psdd_out_dir, num_components)

	print('excuting: {}'.format(cmd))
	os.system(cmd)

def learn_psdd_from_data(train_data_file,valid_data_file, test_data_file, vtree_file, psdd_file, psdd_out_dir):
	cmd = 'java -jar {} learnPsdd search -d {} -b {} -t {} -v {} -m l-1 -p {} -o {}'.format(\
		LEARNPSDD_CMD, train_data_file, valid_data_file, test_data_file, vtree_file, psdd_file,psdd_out_dir)

	print('excuting: {}'.format(cmd))
	os.system(cmd)

def do_training(experiment_dir,cluster_name):
	os.system('pwd')
	small_data_set = True

	# experiment_name = 'ex_4_emnist_32_8'
	# cluster_name = 'james10'
	# dataset = 'mnist'

	encoded_data_dir = os.path.join(experiment_dir,'encoded_data')

	# learn_encoder(testing = testing)
	encode_data(experiment_dir.split('/')[-1], testing = small_data_set)
	return

	symbolic_dir = os.path.join(experiment_dir, 'symbolic_stuff_{}/'.format(cluster_name))
	opt_file = os.path.join(experiment_dir, 'opt.txt')
	vtree_file_learned = os.path.join(symbolic_dir, '{}_learned.vtree'.format('model'))#experiment_name))
	vtree_file_compiled = os.path.join(symbolic_dir, '{}_compiled.vtree'.format('model'))#experiment_name))
	sdd_file_lvt = os.path.join(symbolic_dir, 'constrains_lvt.sdd')#.format('model'))#experiment_name))
	sdd_file_cvt = os.path.join(symbolic_dir, 'constrains_cvt.sdd')#.format('model'))#experiment_name))
	psdd_file_cvt = os.path.join(symbolic_dir, 'constrains_cvt.psdd')#.format('model'))#experiment_name))
	psdd_file_lvt = os.path.join(symbolic_dir, 'constrains_lvt.psdd')#.format('model'))#experiment_name))
	psdd_out_dir = os.path.join(experiment_dir, 'psdd_model/')
	psdd_ens_out_dir = os.path.join(experiment_dir, 'psdd_model_{}/'.format(cluster_name))


	for root, dir_names, file_names in os.walk(encoded_data_dir):
		for i in file_names:
			if 'train.data' in i:
				train_data_file = os.path.join(root, i)
			# elif 'valid.data' in i:
			# 	valid_data_file = os.path.join(root, i)
			# elif 'test.data' in i:
			# 	test_data_file = os.path.join(root, i)

	# with open(train_data_file, 'r') as f:
	# 	for line in f:
	# 		total_num_variables = len(line.split(','))
	# 		break

	if not os.path.exists(symbolic_dir):
		os.mkdir(symbolic_dir)

	#Make vtree (from data or constraints) make sdd from contraints
	learn_vtree(train_data_file, vtree_file_learned)

	# # compile_constraints_to_sdd(opt_file, sdd_file_cvt, vtree_file_compiled, total_num_variables, symbolic_dir, precomputed_vtree = False)
	# # compile_constraints_to_sdd(opt_file, sdd_file_lvt, vtree_file_learned, total_num_variables, symbolic_dir, precomputed_vtree = True)
	

	# # compile_sdd_to_psdd(train_data_file, valid_data_file, test_data_file, vtree_file_compiled, sdd_file_cvt, psdd_file_cvt)
	# # compile_sdd_to_psdd(train_data_file, valid_data_file, test_data_file, vtree_file_learned, sdd_file_lvt, psdd_file_lvt)
	# # if not os.path.exists(psdd_out_dir):
	# # 	os.mkdir(psdd_out_dir)
	# # learn_psdd_from_data(train_data_file, valid_data_file, test_data_file, vtree_file_compiled, psdd_file_cvt, psdd_out_dir)

	if not os.path.exists(psdd_ens_out_dir):
		os.mkdir(psdd_ens_out_dir)

	dataDir = train_data_file.replace('train.data','')
	learn_ensembly_psdd_2_from_data(dataDir, vtree_file_learned, psdd_ens_out_dir, num_components = 10)


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

if __name__ == '__main__':
	experiment_name = 'ex_1_fl16_c2'
	cluster_id = 'student_compute'

	experiment_dir = os.path.abspath('../output/experiments/{}/'.format(experiment_name))

	do_training(experiment_dir, cluster_id)
	# measure_classifcation_acc(experiment_dir, cluster_id, test = False)
	# draw_class_samples(experiment_dir, cluster_id)
	# decode_class_samples(experiment_dir, cluster_id)




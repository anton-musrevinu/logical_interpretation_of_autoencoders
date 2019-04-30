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

def encode_data(testing = False):
	if testing:
		os.system('python {} --phase encode --experiment_name {} --limit_conversion 100'.format(LOWLEVEL_CMD, experiment_name))
	else:
		os.system('python {} --phase encode --experiment_name {}'.format(LOWLEVEL_CMD, experiment_name))

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

if __name__ == '__main__':
	os.system('pwd')
	small_data_set = False

	experiment_name = 'ex_6_emnist_32_4_staff'
	# dataset = 'mnist'

	experiment_dir = os.path.abspath('../output/experiments/{}/'.format(experiment_name))
	encoded_data_dir = os.path.join(experiment_dir,'encoded_data')

	# learn_encoder(testing = testing)
	encode_data(testing = small_data_set)

	symbolic_dir = os.path.join(experiment_dir, 'symbolic_stuff/')
	opt_file = os.path.join(experiment_dir, 'opt.txt')
	vtree_file_learned = os.path.join(symbolic_dir, '{}_learned.vtree'.format('model'))#experiment_name))
	vtree_file_compiled = os.path.join(symbolic_dir, '{}_compiled.vtree'.format('model'))#experiment_name))
	sdd_file_lvt = os.path.join(symbolic_dir, 'constrains_lvt.sdd')#.format('model'))#experiment_name))
	sdd_file_cvt = os.path.join(symbolic_dir, 'constrains_cvt.sdd')#.format('model'))#experiment_name))
	psdd_file_cvt = os.path.join(symbolic_dir, 'constrains_cvt.psdd')#.format('model'))#experiment_name))
	psdd_file_lvt = os.path.join(symbolic_dir, 'constrains_lvt.psdd')#.format('model'))#experiment_name))
	psdd_out_dir = os.path.join(experiment_dir, 'psdd_model/')
	psdd_ens_out_dir = os.path.join(experiment_dir, 'ensembly_psdd_model/')


	for root, dir_names, file_names in os.walk(encoded_data_dir):
		for i in file_names:
			if 'train.data' in i:
				train_data_file = os.path.join(root, i)
			elif 'valid.data' in i:
				valid_data_file = os.path.join(root, i)
			elif 'test.data' in i:
				test_data_file = os.path.join(root, i)

	with open(train_data_file, 'r') as f:
		for line in f:
			total_num_variables = len(line.split(','))
			break

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




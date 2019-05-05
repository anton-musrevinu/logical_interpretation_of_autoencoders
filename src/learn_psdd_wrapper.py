import os, platform, shutil

# function that have to be at beginning of the scipt

def write(message, level = 'info'):
	out_string = '[{}]\t- {}'.format(level.upper(), message)
	if level == 'error':
		print(out_string)
		raise Exception(out_string)
	elif level == 'cmd-start':
		out_string = '\n{}\n'.format(out_string)
		out_string += '-'* 15 + ' CMD OUTPUT ' + '-'*15
	elif level == 'cmd-end':
		out_string = '=' * 15 + 'CMD OUTPUT END' + '=' * 15 + '\n' + out_string

	print(out_string)

def _check_if_file_exists(file_path, raiseException = True):
	if os.path.isfile(file_path):
		return True
	elif raiseException:
		write('Trying to use file that does not exist: {}'.format(file_path), 'error')
	else:
		return False

def _check_if_dir_exists(dir_path, raiseException = True):
	if os.path.isdir(dir_path):
		return True
	elif raiseException:
		write('Trying to use directory that does not exist: {}'.format(dir_path), 'error')
	else:
		return False

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================

#DEPENDENCIES:

# - PYTHON 3.+

# - Scala-PlearnPsdd 	(STARAI-UCLA software)  -   Link: https://github.com/YitaoLiang/Scala-LearnPsdd
# 													The root location of the source directory should be specified (relative to home, or abs) in the following variable
LEARNPSDD_ROOT_DIR = os.path.join(os.environ['HOME'],'./code/msc/src/Scala-LearnPsdd/')
LEARNPSDD_ROOT_DIR = os.path.abspath(LEARNPSDD_ROOT_DIR)
write('LEARNPSDD_ROOT_DIR'.format(LEARNPSDD_ROOT_DIR),'init')
_check_if_dir_exists(LEARNPSDD_ROOT_DIR)
LEARNPSDD_CMD = os.path.abspath(os.path.join(LEARNPSDD_ROOT_DIR,'./target/scala-2.11/psdd.jar'))
LEARNPSDD_LIB = os.path.abspath(os.path.join(LEARNPSDD_ROOT_DIR, './lib/')) + '/'
_check_if_file_exists(LEARNPSDD_CMD)
_check_if_dir_exists(LEARNPSDD_LIB)
# -------------------------------------------------------------------------------------------------------------------------
#
# - GRAPHVIZ   			(graphing software)     -   Without this please specify:
#
GRAPHVIZ_INSTALLED = True
# -------------------------------------------------------------------------------------------------------------------------
#
# - SDDLIB BINARY       (STAR-UCLA software)    -   Link: http://reasoning.cs.ucla.edu/sdd/
#
SDDLIB_BIN = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/src/wmisdd/bin/'))
write('SDDLIB_BIN'.format(SDDLIB_BIN),'init')
_check_if_dir_exists(SDDLIB_BIN)
if 'Linux' in platform.system():
	SDDLIB_CMD = os.path.abspath(os.path.join(SDDLIB_BIN, 'sdd-linux'))
else:
	SDDLIB_CMD = os.path.abspath(os.path.join(SDDLIB_BIN, 'sdd-darwin'))
	write('the program only works fully on linux based systems, so some aspects might not work for you\n --> Assuming OSX', 'warning')
write('SDDLIB_CMD'.format(SDDLIB_CMD),'init')
_check_if_file_exists(SDDLIB_BIN)
# -------------------------------------------------------------------------------------------------------------------------
#
# - Updated Source version of LearnPSDD (STAR-UCLA Software) - Link:
#
LEARNPSDD2_CMD = os.path.abspath(os.path.join(LEARNPSDD_ROOT_DIR,'../learnPSDD/target/scala-2.11/psdd.jar'))
write('LEARNPSDD2_CMD'.format(LEARNPSDD2_CMD),'init')
_check_if_file_exists(LEARNPSDD2_CMD)



#============================================================================================================================
#============================================ AUXILIARY FUNCTIONS ====================================================
#============================================================================================================================


def convert_dot_to_pdf(file_path, do_this = True):
	if not do_this or not _check_if_file_exists(file_path + '.dot', raiseException = False) or not GRAPHVIZ_INSTALLED:
		return

	cmd_str = 'dot -Tpdf {}.dot -o {}.pdf'.format(file_path,file_path)
	os.system(cmd_str)
	write('Converted file to pdf (graphical depictoin). Location: {}'.format(file_path + '.pdf'))

def add_learn_psdd_lib_to_path():
	# os.environ['LD_LIBRARY_PATH'] = ''
	if 'LD_LIBRARY_PATH' not in os.environ:
		os.environ['LD_LIBRARY_PATH'] = LEARNPSDD_LIB# + os.pathsep + SDD_LIB_DIR
		write('variable LD_LIBRARY_PATH created and set to: {}'.format(os.environ['LD_LIBRARY_PATH']))
	
	if not LEARNPSDD_LIB in os.environ['LD_LIBRARY_PATH']:
		os.environ['LD_LIBRARY_PATH'] += os.pathsep + LEARNPSDD_LIB
		write('variable LD_LIBRARY_PATH updated to: {}'.format(os.environ['LD_LIBRARY_PATH']))
	
	# if not SDD_LIB_DIR in os.environ['LD_LIBRARY_PATH']:
	# 	os.environ['LD_LIBRARY_PATH'] += os.pathsep + SDD_LIB_DIR
	# 	write('variable LD_LIBRARY_PATH updated to: {}'.format(os.environ['LD_LIBRARY_PATH']))

	if not LEARNPSDD_LIB in os.environ['PATH']:
		os.environ['PATH'] += str(os.pathsep + LEARNPSDD_LIB)
		write('variable PATH updated to: {}'.format(os.environ['PATH']))

	write(os.environ['LD_LIBRARY_PATH'])

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================

def learn_vtree(train_data_path, vtree_path, vtree_method = 'miBlossom', convert_to_pdf = True):
	'''
	-d, --trainData <file>   
	-v, --vtreeMethod leftLinear-rand|leftLinea-ord|pairwiseWeights|balanced-rand|rightLinear-ord|rightLinear-rand|balanced-ord|miBlossom|miGreedyBU|miMetis
						   default: miBlossom
		 * leftLinear-rand: left linear vtree a random order
		 * leftLinea-ord: left linear vtree using variable order
		 * pairwiseWeights: balanced vtree by top down selection of the split that minimizes the mutual information between the two parts, using exhaustive search.
		 * balanced-rand: balanced vtree using a random order
		 * rightLinear-ord: right linear vtree using variable order
		 * rightLinear-rand: right linear vtree using a random order
		 * balanced-ord: balanced vtree using variable order
		 * miBlossom: balanced vtree by bottom up matching of the pairs that maximizes the average mutual information in a pair, using blossomV.
		 * miGreedyBU: balanced vtree by bottom up matching of the pairs that maximizes the average mutual information in a pair, using greedy selection.
		 * miMetis: balanced vtree by top down selection of the split that minimizes the mutual information between the two parts, using metis.
	-o, --out <path>         
	-e, --entropyOrder       choose prime variables to have lower entropy
	'''
	_check_if_file_exists(train_data_path)


	cmd_str = 'java -jar {} learnVtree '.format(LEARNPSDD_CMD) + \
		  ' --trainData {}'.format(train_data_path) + \
		  ' --vtreeMethod {}'.format(vtree_method) + \
		  ' --out {}'.format(vtree_path.replace('.vtree',''))


	add_learn_psdd_lib_to_path()

	write(cmd_str,'cmd-start')
	os.system(cmd_str)

	_check_if_file_exists(vtree_path)
	
	write('Finished leraning Vtree from data. File location: {}'.format(vtree_path), 'cmd-end')

	convert_dot_to_pdf(vtree_path, convert_to_pdf)

def compile_cnf_to_sdd(cnf_path, sdd_path, vtree_out_path, vtree_in_path = None, initial_vtree_type = 'random', vtree_search_freq = 5, post_compilation_vtree_search = True, convert_to_pdf = True):
	'''
	If vtree_in_path == None, invoke vtree search every k clauses

	libsdd version 2.0, January 08, 2018
	sdd (-c FILE | -d FILE | -s FILE) [-v FILE | -t TYPE] [-WVRS FILE] [-r MODE] [-mhqp]
	  -c FILE         set input CNF file
	  -d FILE         set input DNF file
	  -v FILE         set input VTREE file
	  -s FILE         set input SDD file
	  -W FILE         set output VTREE file
	  -V FILE         set output VTREE (dot) file
	  -R FILE         set output SDD file
	  -S FILE         set output SDD (dot) file
	  -m              minimize the cardinality of compiled sdd
	  -t TYPE         set initial vtree type (left/right/vertical/balanced/random)
	  -r K            if K>0: invoke vtree search every K clauses
						if K=0: disable vtree search
						by default (no -r option), dynamic vtree search is enabled
	  -q              perform post-compilation vtree search
	  -h              print this help and exit
	  -p              verbose output
	  '''

	_check_if_file_exists(cnf_path)
	_check_if_file_exists(vtree_out_path)

	generate_vtree = vtree_in_path == None

	if not generate_vtree:
		_check_if_file_exists(vtree_in_path)

	cmd_str = '{} '.format(SDDLIB_CMD) + \
		  ' -c {}'.format(cnf_path)
	
	if generate_vtree:
		cmd_str += ' -t {}'.format(initial_vtree_type)
	else:
		cmd_str += ' -v {}'.format(vtree_in_path)

	cmd_str += ' -W {}'.format(vtree_out_path)
	cmd_str += ' -V {}.dot'.format(vtree_out_path)
	cmd_str += ' -R {}'.format(sdd_path)
	cmd_str += ' -S {}.dot'.format(sdd_path)
	cmd_str += ' -m'

	if generate_vtree:
		cmd_str += ' -r {}'.format(vtree_search_freq)
	else:
		cmd_str += ' -r 0'

	if post_compilation_vtree_search:
		cmd_str += ' -q'

	add_learn_psdd_lib_to_path()
	write(cmd_str,'cmd-start')
	os.system(cmd_str)

	_check_if_file_exists(vtree_out_path) and _check_if_file_exists(sdd_path)
	write('Finished compiling CNF to SDD. File location: {}'.format(sdd_path), 'cmd-end')

	convert_dot_to_pdf(vtree_out_path,convert_to_pdf)
	convert_dot_to_pdf(sdd_path,convert_to_pdf)

def compile_sdd_to_psdd(train_data_path, vtree_path, sdd_path, psdd_path, valid_data_path = None, test_data_path = None, smoothing = 'l-1'):
	'''
	  -d, --trainData <file>   
	  -b, --validData <file>   
	  -t, --testData <file>    
	  -v, --vtree <file>       
	  -s, --sdd <file>         
	  -o, --out <path>         
	  -m, --smooth <smoothingType>
							   default: l-1
			 * mc-m-<m>: m-estimator, weighted with model count
			 * mc-<m>: model count as pseudo count, weighted with m
			 * l-<m>: laplace smoothing, weighted with m
			 * m-<m>: m-estimator smoothing
			 * mc-l-<m>: laplace smoothing, weighted with model count and m
			 * no: No smoothing
	  --help                   prints this usage text
	'''
	_check_if_file_exists(train_data_path)
	_check_if_file_exists(vtree_path)
	_check_if_file_exists(sdd_path)

	cmd_str = 'java -jar {} sdd2psdd '.format(LEARNPSDD_CMD) + \
		  ' -d {}'.format(train_data_path) + \
		  ' -v {}'.format(vtree_path) + \
		  ' -s {}'.format(sdd_path) + \
		  ' -o {}'.format(psdd_path.replace('.psdd', '')) + \
		  ' -m {}'.format(smoothing)

	if valid_data_path != None and _check_if_file_exists(valid_data_path, raiseException = False):
		cmd_str += ' -b {}'.format(valid_data_path)
	if test_data_path != None and _check_if_file_exists(test_data_path, raiseException = False):
		cmd_str += ' -t {}'.format(test_data_path)


	add_learn_psdd_lib_to_path()
	write(cmd_str,'cmd-start')
	os.system(cmd_str)

	_check_if_file_exists(psdd_path)
	write('Finished compiling SDD to PSDD. File location: {}'.format(psdd_path), 'cmd-end')

def learn_psdd_from_data(train_data_path, vtree_path, output_dir, psdd_input_path = None, 
		valid_data_path = None, test_data_path = None, smoothing = 'l-1', clone_k = 3, split_k = 1, completion = 'maxDepth-3', scorer = 'dll/ds',
		maxIt = 'max', save_freq = 'best-3'):
	'''
	  -p, --psdd <file>        If no psdd is provided, the learning starts from a mixture of marginals.
	  -v, --vtree <file>
	  -d, --trainData <file>   
	  -b, --validData <file>   
	  -t, --testData <file>
	  -o, --out <path>         The folder for output

	  -m, --smooth <smoothingType>
							   default: l-1
			 * mc-m-<m>: m-estimator, weighted with model count
			 * mc-<m>: model count as pseudo count, weighted with m
			 * l-<m>: laplace smoothing, weighted with m
			 * m-<m>: m-estimator smoothing
			 * mc-l-<m>: laplace smoothing, weighted with model count and m
			 * no: No smoothing

					 * l-<m>: laplace smoothing, weighted with m
			 * m-<m>: m-estimator smoothing
			 * mc-l-<m>: laplace smoothing, weighted with model count and m
			 * no: No smoothing

	  -h, --opTypes <opType>,<opType>,...
							   default: clone-3,split-1
			options: clone-<k>,split-<k>
			In split-k, k is the number of splits.
			In clone-k, k is the maximum number of parents to redirect to the clone.

	  -c, --completion <completionType>
							   default: maxDepth-3
			 * complete
			 * min
			 * maxDepth-<k>
			 * maxEdges-<k>

	  -s, --scorer <scorer>    default: dll/ds
			 * dll
			 * dll/ds

	  -e, --maxIt <maxIt>      For search, this is the maximum number of operations to be applied on the psdd.
			For bottom-up and top-down, at every level at most f*#vtreeNodesAtThisLevel operations will be applied.
			default: maxInt

	  -f, --freq <freq>        method for saving psdds
			default: best-1
			 * all-<k>
			 * best-<k>
			best-k to only keep the best psdd on disk, all-k keeps all of the. A save attempt is made every k iterations

	  -q, --debugLevel <level>
							   debug level

	  --help                   prints this usage text
	'''
	_check_if_file_exists(train_data_path)
	_check_if_file_exists(vtree_path)
	_check_if_dir_exists(output_dir)

	cmd_str = 'java -jar {} learnPsdd search '.format(LEARNPSDD_CMD) + \
		  ' --trainData {}'.format(train_data_path) + \
		  ' --vtree {}'.format(vtree_path) + \
		  ' --out {}'.format(output_dir)

	if valid_data_path != None and _check_if_file_exists(valid_data_path, raiseException = False):
		cmd_str += ' --validData {}'.format(valid_data_path)
	if test_data_path != None and _check_if_file_exists(test_data_path, raiseException = False):
		cmd_str += ' --testData {}'.format(test_data_path)
	if psdd_input_path != None and _check_if_file_exists(psdd_input_path, raiseException = False):
		cmd_str += ' -p {}'.format(psdd_input_path)

	cmd_str += ' --smooth {}'.format(smoothing) + \
			   ' --opTypes clone-{},split-{}'.format(clone_k, split_k) + \
			   ' --completion {}'.format(completion) + \
			   ' --scorer {}'.format(scorer) + \
			   ' --freq {}'.format(save_freq)
	if not maxIt == 'max':
		cmd_str += ' --maxIt {}'.format(maxIt)


	add_learn_psdd_lib_to_path()
	write(cmd_str,'cmd')
	os.system(cmd_str)

	final_psdd_file = os.path.join(psdd_learner_tmp_dir,'./models/final.psdd')
	_check_if_file_exists(final_psdd_file)

	write('Finished PSDD learnin. File location: {}'.format(final_psdd_file), 'cmd-end')
	# if _check_if_file_exists(psdd_path, raiseException = False):
	# 	write('Finished compiling SDD to PSDD. File location: {}'.format(psdd_path))


def learn_ensembly_psdd_from_data(train_data_path, vtree_path, output_dir, psdd_input_path = None, num_compent_learners = 5, 
		valid_data_path = None, test_data_path = None, smoothing = 'l-1', structureChangeIt = 3, parameterLearningIt = 1, scorer = 'dll/ds',
		maxIt = 'maxInt', save_freq = 'best-3'):
	'''
	  -p, --psdd <file>        If no psdd is provided, the learning starts from a mixture of marginals.

	  -v, --vtree <file>       
	  -d, --trainData <file>   
	  -b, --validData <file>   
	  -t, --testData <file>    
	  -o, --out <path>         The folder for output

	  -c, --numComponentLearners <numComponentLearners>
							   The number of component learners to form the ensemble

	  -m, --smooth <smoothingType>
							   default: l-1
			 * mc-m-<m>: m-estimator, weighted with model count
			 * mc-<m>: model count as pseudo count, weighted with m
			 * l-<m>: laplace smoothing, weighted with m
			 * m-<m>: m-estimator smoothing
			 * mc-l-<m>: laplace smoothing, weighted with model count and m
			 * no: No smoothing

	  -s, --scorer <scorer>    default: dll/ds
			 * dll
			 * dll/ds

	  -e, --maxIt <maxIt>      this is the maximum number of ensemble learning iterations.

	  --structureChangeIt <structureChangeIt>
							   this is the number of structure changes before a new round of parameter learning .

	  --parameterLearningIt <parameterLearningIt>
							   this is the number of iterators for parameter learning before the structure of psdds changes again.

	  --help                   prints this usage text
	'''

	_check_if_file_exists(train_data_path)
	_check_if_file_exists(vtree_path)
	_check_if_dir_exists(output_dir)

	cmd_str = 'java -jar {} learnEnsemblePsdd softEM '.format(LEARNPSDD_CMD) + \
		  ' --trainData {}'.format(train_data_path) + \
		  ' --vtree {}'.format(vtree_path) + \
		  ' --out {}'.format(output_dir) + \
		  ' --numComponentLearners {}'.format(num_compent_learners)

	if valid_data_path != None and _check_if_file_exists(valid_data_path, raiseException = False):
		cmd_str += ' --validData {}'.format(valid_data_path)
	if test_data_path != None and _check_if_file_exists(test_data_path, raiseException = False):
		cmd_str += ' --testData {}'.format(test_data_path)
	if psdd_input_path != None and _check_if_file_exists(psdd_input_path, raiseException = False):
		cmd_str += ' -p {}'.format(psdd_input_path)

	cmd_str += ' --smooth {}'.format(smoothing) + \
			   ' --opTypes clone-{},split-{}'.format(clone_k, split_k) + \
			   ' --completion {}'.format(completion) + \
			   ' --scorer {}'.format(scorer) + \
			   ' --maxIt {}'.format(maxIt)# + \
			   # ' --freq {}'.format(save_freq)


	add_learn_psdd_lib_to_path()
	write(cmd_str,'cmd')
	os.system(cmd_str)

	# cmd = 'java -jar {} learnEnsemblePsdd softEM -d {} -b {} -t {} -v {} -m l-1 -o {} -c {}'.format(\
	# 	LEARNPSDD_CMD, train_data_file, valid_data_file, test_data_file, vtree_file,psdd_out_dir,num_components)

	# print('excuting: {}'.format(cmd))
	# os.system(cmd)

def learn_ensembly_psdd2_from_data(dataDir, vtree_path, output_dir, num_components = 10):

	# Method for running the psdd code from the paper with (updated source)

	_check_if_file_exists(dataDir + 'train.data')
	_check_if_file_exists(dataDir + 'valid.data')
	_check_if_file_exists(dataDir + 'test.data')
	_check_if_file_exists(vtree_path)
	_check_if_dir_exists(output_dir)

	cmd_str = 'java -jar {} SoftEM {} {} {} {}'.format(\
		LEARNPSDD2_CMD, dataDir, vtree_file, psdd_out_dir, num_components)

	print('excuting: {}'.format(cmd_str))
	os.system(cmd_str)


# ====================================================================================================================
# ========================================   Higher level methods    =================================================
# ====================================================================================================================

def learn_psdd(experiment_name, train_data_path, 
		experiment_dir_path = './experiments/', valid_data_path = None, test_data_path = None, \
		replace_existing = False, vtree_method = 'miBlossom', num_compent_learners = 1, psdd_in_file = None, \
		constraints_cnf_file = None, keep_generated_files = True, convert_to_pdf = True):

	experiment_path = os.path.join(experiment_dir_path, experiment_name)
	write('experiment dir path: {}'.format(experiment_path),'files')
	if _check_if_dir_exists(experiment_path, raiseException = False):
		if not replace_existing:
			write('Experiment dir already exists, please delete this or specify to do so in the arguments. (or give a different name)', 'error')
		else:
			shutil.rmtree(experiment_path)
	os.mkdir(experiment_path)

	psdd_learner_tmp_dir = os.path.join(experiment_path, './psdd_learner/')
	os.mkdir(psdd_learner_tmp_dir)
	write('psdd learner dir: {}'.format(psdd_learner_tmp_dir), 'files')
	output_vtree_file = os.path.join(experiment_path, './model.vtree')
	write('output vtree file: {}'.format(output_vtree_file), 'files')
	output_psdd_file = os.path.join(experiment_path, './model.psdd')
	write('output psdd dir: {}'.format(output_psdd_file), 'files')
	contraints_tmp_dir = None

	#Learn vtree from data 
	learn_vtree(train_data_path, output_vtree_file, vtree_method = vtree_method, convert_to_pdf = convert_to_pdf)

	#Compile constraints to sdd/psdd if present
	if constraints_cnf_file != None and psdd_in_file == None:
		_check_if_file_exists(constraints_cnf_file, raiseException = True)

		contraints_tmp_dir = os.path.join(experiment_dir, './constraints_tmp/')
		os.mkdir(contraints_tmp_dir)

		constraints_sdd_file = os.path.join(contraints_tmp_dir, './constraints_as.sdd')
		compile_cnf_to_sdd(constraints_cnf_file, constraints_sdd_file, output_vtree_file, \
							vtree_in_path = output_vtree_file, post_compilation_vtree_search = False, convert_to_pdf = convert_to_pdf)
		
		constraints_psdd_file = os.path.join(contraints_tmp_dir, './constraints_as.psdd')
		compile_sdd_to_psdd(train_data_path, output_vtree_file, sdd_path, psdd_path, \
							valid_data_path = valid_data_path, test_data_path = test_data_path)

		psdd_in_file = constraints_psdd_file

	if num_compent_learners < 1:
		write('The number of psdd_compents (for ensembly learning) has to be > 0', 'error')
	elif num_compent_learners == 1:
		learn_psdd_from_data(train_data_path, output_vtree_file, psdd_learner_tmp_dir, valid_data_path = valid_data_path, \
			test_data_path = test_data_path, psdd_input_path = psdd_in_file)
	else:
		raise Expection('not yet supported')
		# learn_ensembly_psdd_from_data(train_data_path, output_vtree_file, psdd_learner_tmp_dir,psdd_input_path = psdd_in_file\
		# 	 num_compent_learners = num_compent_learners, valid_data_path = valid_data_path, test_data_path = test_data_path)

	final_psdd_file = os.path.join(psdd_learner_tmp_dir,'./models/final.psdd')
	if not _check_if_file_exists(final_psdd_file, raiseException = False):
		write('final psdd file counld not be found at location: {}'.format(final_psdd_file),'error')	
	shutil.copyfile(final_psdd_file, output_psdd_file)

	final_psdd_dot_file = os.path.join(psdd_learner_tmp_dir,'./models/final.dot')
	if not _check_if_file_exists(final_psdd_dot_file, raiseException = False):
		write('final psdd dot file counld not be found at location: {}'.format(final_psdd_dot_file),'warning')
	else:
		shutil.copyfile(final_psdd_dot_file, output_psdd_file + '.dot')
		convert_dot_to_pdf(output_psdd_file, convert_to_pdf)

	if not keep_generated_files:
		if _check_if_dir_exists(psdd_learner_tmp_dir, raiseException = False):
			shutil.rmtree(psdd_learner_tmp_dir)
		if _check_if_dir_exists(contraints_tmp_dir, raiseException = False):
			shutil.rmtree(contraints_tmp_dir)

	write('Program finished.','final')


# ============================================================================================================================
# ============================================================================================================================
# ============================================================================================================================


if __name__ == '__main__':
	experiment_dir = os.path.join(os.environ['HOME'],'./code/msc/output/experiments/ex_1_fl16_c2')
	experiment_name = 'psdd_search_v0'
	experiment_dir_path = os.path.abspath(experiment_dir)
	test_data_path = os.path.join(experiment_dir_path, 'encoded_data/mnist-encoded-valid_MSE-test.data')
	valid_data_path = os.path.join(experiment_dir_path, 'encoded_data/mnist-encoded-valid_MSE-valid.data')
	train_data_path = os.path.join(experiment_dir_path, 'encoded_data/mnist-encoded-valid_MSE-train.data')

	learn_psdd(experiment_name, train_data_path, experiment_dir_path, replace_existing = True)

	# def learn_psdd(experiment_name, train_data_path, 
	# 	experiment_dir_path = './experiments/', valid_data_path = None, test_data_path = None, \
	# 	replace_existing = False, vtree_method = 'miBlossom', num_compent_learners = 1, psdd_in_file = None, \
	# 	constraints_cnf_file = None, keep_generated_files = True, convert_to_pdf = True):


def do_training(experiment_dir,cluster_name):
	os.system('pwd')
	small_data_set = False

	# experiment_name = 'ex_4_emnist_32_8'
	# cluster_name = 'james10'
	# dataset = 'mnist'

	encoded_data_dir = os.path.join(experiment_dir,'encoded_data')

	# learn_encoder(testing = testing)
	encode_data(experiment_dir.split('/')[-1], testing = small_data_set)

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






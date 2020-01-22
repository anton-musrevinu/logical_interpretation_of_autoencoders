import os
import numpy as np
import csv


RELATIVE_BASEDIR = './code/msc/output/experiments/'
RELATIVE_OUTFILE_all = './code/msc/output/var_results_all_mnist.csv'
RELATIVE_BASEDIR_all = './local_storage/backup_msc/output/experiments/'
BASEDIR = os.path.abspath(os.path.join(os.environ['HOME'],RELATIVE_BASEDIR))
BLOODBORN_BASE = os.path.abspath(os.path.join(os.environ['HOME'],'./local_storage/backup_msc/output/experiments/'))
OUTFILE_var_mnist = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/var_results_mnist.csv'))
OUTFILE_var_emnist = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/var_results_emnist.csv'))
OUTFILE_all = os.path.abspath(os.path.join(os.environ['HOME'],RELATIVE_OUTFILE_all))

def read_opt_file(dir):
	flx_size = None
	fl_cat_dim = None
	with open(os.path.join(dir, './opt.txt')) as f:
		for line in f:
			if not ':' in line:
				continue
			identifier = line.split(':')[0].repace(' ', '')
			value = line.spit(':')[1]
			if '[' in line:
				value = value.split('[')[0].replace(' ', '')
			else:
				value = value.replace('\n', '').replace(' ', '')
			if identifier == 'feature_layer_size':
				flx_size = int(value)
			elif identifier == 'categorical_dim':
				fl_cat_dim = int(value)
			else:
				continue
	return flx_size, fl_cat_dim

def read_opt_file_for_key(dir, key):
	return_value = None
	with open(os.path.join(dir, './opt.txt')) as f:
		for line in f:
			if not ':' in line:
				continue
			identifier = line.split(':')[0].repace(' ', '')
			value = line.spit(':')[1]
			if '[' in line:
				value = value.split('[')[0].replace(' ', '')
			else:
				value = value.replace('\n', '').replace(' ', '')
			if identifier == key:
				return_value = value
			else:
				continue
	return return_value


def get_task_type_hiracy(task_type_of_intersest):
	if task_type_of_intersest == 'classification':
		return lambda x: x == 'classification'
	elif task_type_of_intersest == 'compositional':
		return lambda x: x != 'classification' and not 'noisy' in x and not 'succ' in x
	elif task_type_of_intersest == 'noisy':
		return lambda x: 'noisy' in x

class ExpResult(object):
	def __init__(self,flx_size, flx_cat_dim):
		self.flx_size = flx_size
		self.flx_cat_dim = flx_cat_dim
		self.complexity_num = flx_cat_dim ** flx_size
		self.complexity_bin = int(np.ceil(np.log2(self.flx_cat_dim))) * self.flx_size

	def add_dataset(self, dir)
		self.dataset = read_opt_file_for_key(dir, 'dataset')

	def add_loss(self,loss):
		self.loss = loss

	def add_losses_from_dir(self,dir):
		summary = os.path.abspath(os.path.join(dir, './VAEManager/summary_training.txt'))
		with open(summary, 'r') as f:
			for_right_error = False
			for line in f:
				if line.startswith('for error: valid_MSE'):
					for_right_error = True
				elif line.startswith('for error:'):
					for_right_error = False

				if for_right_error and (line.strip().startswith('valid_MSE:') or line.strip().startswith('corresponding value:')):
					loss = float(line.strip().split(':')[-1])
		self.loss = loss

		# print('bin complexity: {} - loss found: {}'.format(self.complexity_bin,self.loss))

class ExpResultFullFile(ExpResultFull):
	def __init__(self, dir):
		flx_size, flx_cat_dim = read_opt_file(dir)
		ExpResultFull.__init__(self, flx_size, flx_cat_dim)


class ExpResultFull(ExpResult):
	def __init__(self, flx_size, flx_cat_dim):
		ExpResult.__init__(self, flx_size, flx_cat_dim)
		self.exp_psdds = {}

	def __str__(self):
		outstr = 'EXP: fl_size: {}, fl_cat_dim: {}, recLoss: {}   (bin complexity: {})'.format(self.flx_size,self.flx_cat_dim, self.loss, self.complexity_bin)
		for i in self.exp_psdds.keys():
			outstr += '\n' + self.exp_psdds[i].__str__(offset = '\t')
		return outstr

	def add_psdd_exp(self, dir):
		for experiment_dir_path, dir_names_2, file_names_2 in os.walk(dir):
			for psdd_search_dir in dir_names_2:
				for psdd_dir_name in ['psdd_search', 'psdd_model']:
					if psdd_dir_name in psdd_search_dir:
						identifier = str(psdd_search_dir).split(psdd_dir_name + '_')[1].replace('/','')
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

						if not task_type in self.exp_psdds:
							self.exp_psdds[task_type] = ExpPsdd(task_type)

						self.exp_psdds[task_type].get_best_ll_from_progress(progressfile, cluster_id)
						self.exp_psdds[task_type].add_fly_info(experiment_dir_path, cluster_id)

						vtree_folder = os.path.join(exp_cluster_dir, './learnvtree_tmp_dir')
						self.exp_psdds[task_type].add_vtree_method(cluster_id, vtree_folder)

						evaluationDir = os.path.abspath(os.path.join(exp_cluster_dir, './evaluation'))
						if not os.path.exists(evaluationDir):
							continue
							# print('added because evaldir does not exist', evaluationDir)
						else:
							for root_3, dir_names_3, file_names_3 in os.walk(evaluationDir):
								if not any(['classification' in file_name for file_name in file_names_3]):
									continue
								else:
									for file_4 in file_names_3:
										if 'classification' in file_4 and file_4.endswith('.info') and os.path.getsize(os.path.join(root_3, file_4)) != 0:
											self.exp_psdds[task_type].extract_and_add_acc_from_file(os.path.join(root_3, file_4), cluster_id)

	# def add_class_acc_from_dir(self,dir, task_type_of_intersest):

	# 	for experiment_dir_path, dir_names_2, file_names_2 in os.walk(dir):
	# 		for psdd_search_dir in dir_names_2:
	# 			if 'psdd_search' in psdd_search_dir:
	# 				identifier = str(psdd_search_dir).split('psdd_search_')[1].replace('/','')
	# 				if len(identifier.split('_')) == 3 or (len(identifier.split('_')) == 2 and 'james' in identifier):
	# 					cluster_id = '_'.join(identifier.split('_')[:-1])
	# 					task_type = identifier.split('_')[-1]
	# 				else:
	# 					cluster_id = identifier
	# 					task_type = 'classification'

	# 				if not get_task_type_hiracy(task_type_of_intersest)(task_type):
	# 					continue

	# 				exp_cluster_dir = os.path.abspath(os.path.join(experiment_dir_path, psdd_search_dir))
	# 				progressfile = os.path.join(exp_cluster_dir, './learnpsdd_tmp_dir/progress.txt')

	# 				if not os.path.exists(progressfile) or os.path.getsize(progressfile) == 0:
	# 					continue

	# 				self._get_best_ll_from_progress(progressfile, (cluster_id, task_type))
	# 				self.add_fly_info(experiment_dir_path, cluster_id, task_type)

	# 				vtree_folder = os.path.join(exp_cluster_dir, './learnvtree_tmp_dir')
	# 				self.add_vtree_method((cluster_id, task_type), vtree_folder)

	# 				evaluationDir = os.path.abspath(os.path.join(exp_cluster_dir, './evaluation'))
	# 				if not os.path.exists(evaluationDir):
	# 					continue
	# 					# print('added because evaldir does not exist', evaluationDir)
	# 				else:
	# 					for root_3, dir_names_3, file_names_3 in os.walk(evaluationDir):
	# 						if not any(['classification' in file_name for file_name in file_names_3]):
	# 							continue
	# 						else:
	# 							allempty = True
	# 							for file_4 in file_names_3:
	# 								if 'classification' in file_4 and file_4.endswith('.info') and os.path.getsize(os.path.join(root_3, file_4)) != 0:
	# 									self._extract_and_add_acc_from_file(os.path.join(root_3, file_4), (cluster_id, task_type))
		
	def add_all_possible_information_for_dir(self,dir):
		self.add_losses_from_dir(dir)
		self.add_psdd_exp(dir)

class ExpPsdd(object):
	def __init__(self, task_type):
		self.task_type = task_type

		self.classification_acc = {}
		self.best_it = {}
		self.best_ll = {}
		self.total_it = {}
		self.vtree_method = {}
		self.compressed_y = {}

	def add_fly_info(self, experiment_dir_path, cluster_id):
		self.compressed_y[cluster_id] = _is_compressed_y(experiment_dir_path, cluster_id, self.task_type)

	def extract_and_add_acc_from_file(self, file_name, identifier):
		with open(file_name, 'r') as f:
			for line in f:
				if line.startswith('The accuracy over all queries is '):
					acc = float(line.split('The accuracy over all queries is ')[1].strip())
					self.add_class_acc(identifier, acc)

	def add_class_acc(self, cluster_id, acc):
		if cluster_id in self.classification_acc.keys():
			self.classification_acc[cluster_id].append(acc)
		else:
			self.classification_acc[cluster_id] = list([acc])

	def add_vtree_method(self, identifier, vtree_folder):
		for root, dir_names, file_names in os.walk(vtree_folder):
			for file in file_names:
				if file.endswith('.vtree'):
					vtree_method = file.split('.')[0].strip()

		self.vtree_method[identifier] = vtree_method

	def get_best_ll_from_progress(self, prgfile, identifier):
		it_weights = {}
		it_ll = {}
		last_it_idx = -2
		with open(prgfile, 'r') as f:
			for line_idx, line in enumerate(f):
				splitted = line.split(';')
				if len(splitted) < 5 or line_idx == 0:
					continue
				num_learners = len(splitted) - 5
				current_it = int(splitted[0].strip())
				current_ll = float(splitted[-1].strip())
				# print(current_ll, current_ll > best_it_ll)
				last_it_idx = current_it
				it_ll[current_it] = current_ll
				it_weights[current_it] = []
				for idx, i in enumerate(range(4,num_learners + 4)):
					it_weights[current_it].append(float(splitted[i].strip()))

		best_iteration = max(it_ll, key=it_ll.get)
		best_ll = it_ll[best_iteration]

		self.add_best_ll_and_iteration(identifier, best_iteration, best_ll, last_it_idx)

	def add_best_ll_and_iteration(self, identifier, best_it, best_ll, total_iterations):
		self.best_it[identifier] = best_it
		self.best_ll[identifier] = best_ll
		self.total_it[identifier] = total_iterations

	def __str__(self, offset = ''):
		outstr = offset + 'TASK TYPE: {}'.format(self.task_type)
		for i in self.classification_acc.keys():
			outstr += '\n' + offset + '\tid:{} - best_it: {}, best_ll: {}, total_it: {}, vtree_method: {}, compressed_y: {}, class_acc: {}'.format(\
				i, self.best_it[i], self.best_ll[i], self.total_it[i], self.vtree_method[i], self.compressed_y[i], self.classification_acc[i])
		return outstr
#==========================================================================================================================
#==========================================================================================================================

def _get_psdd_out_dir(experiment_dir_path, cluster_id, task_type):
	identifier = cluster_id + '_' + task_type if task_type != 'classification' else cluster_id

	if 'ex_5' in experiment_dir_path or 'ex_6' in experiment_dir_path:
		psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_model_{}'.format(identifier))
	else:
		psdd_out_dir = os.path.join(experiment_dir_path,'./psdd_search_{}'.format(identifier))

	return os.path.abspath(psdd_out_dir)

def _is_compressed_y(experiment_dir_path, cluster_id, task_type):
	encoded_data_dir = None

	#Read info file located at tte root of psdd_out_dir
	psdd_out_dir = _get_psdd_out_dir(experiment_dir_path, cluster_id, task_type)
	fl_data_file = os.path.join(psdd_out_dir, './fl_data.info')
	if os.path.exists(fl_data_file):
		identifier = 'encoded_data_dir:'
		with open(fl_data_file, 'r') as f:
			for line in f:
				if line.startswith(identifier):
					relative_path = line.split(identifier)[-1].strip()
					joined = os.path.join(psdd_out_dir, relative_path)
					encoded_data_dir = os.path.abspath(joined)

	#Reconstruct the information in case the file does not exists (old experiemnts)
	if encoded_data_dir == None:
		if 'ex_7_mnist_16_4' in experiment_dir_path and 'james10' == cluster_id or\
			'ex_7_mnist_32_2' in experiment_dir_path and 'student_compute' == cluster_id:
			encoded_data_dir = os.path.join(experiment_dir_path,'encoded_data_uncompressed_y')
		elif task_type != 'classification':
			encoded_data_dir = os.path.join(experiment_dir_path, 'encoded_data_' + task_type)
		else:
			encoded_data_dir = os.path.join(experiment_dir_path,'encoded_data')

	if 'uncompressed_y' in encoded_data_dir:
		return False
	else:
		return True


#==========================================================================================================================
#==========================================================================================================================

def make_var_resutls_file(data = 'ex_7_mnist', sorte_by = lambda x: x.flx_size):
	expResults = gather_only_var_results(data)

	expResultsSorted = sorted(expResults, key=sorte_by, reverse=False)
	if len(expResults) < 1:
		return
	first_dataset = expResults[0].dataset
	for i in expResults:
		if first_dataset != i.dataset:
			raise Exception('not all experiments found have been run on the same dataset, please specify the data attribute better')

	data_type = first_dataset
	if int(data.split('_')[1]) < 6:
		outfile = OUTFILE_var_mnist.replace('mnist', '{}mnist'.format(data))
	else:
		outfile = OUTFILE_var_mnist.replace('mnist', data_type)

	with open(outfile, 'w') as f:
		f.write('dataset, FL categorical size, categorical dimension, MSE, model space complexity\n')
		for exp in expResultsSorted:
			line = '{},\t{},\t{},\t{:.3},\t2^{{{}}}'.format(data_type, exp.flx_size, exp.flx_cat_dim, exp.loss, exp.complexity_bin)
			f.write(line + '\n')

def gather_only_var_results(data):
	experiment_dirs = []
	for root, dir_names, file_names in os.walk(BLOODBORN_BASE):
		for i in dir_names:
			if data in i:
				experiment_dirs.append(os.path.join(root, i))
	
	expResults = []
	for exp_dir in experiment_dirs:
		expResult = ExpResultFullFile(exp_dir)
		expResult.add_dataset(exp_dir)
		expResult.add_losses_from_dir(exp_dir)
		expResults.append(expResult)
	return expResults

def gather_results(data):
	print('\nRetrieving experiments\n')
	experiment_dirs = []
	for root, dir_names, file_names in os.walk(BLOODBORN_BASE):
		for i in dir_names:
			if data in i:
				experiment_dirs.append(os.path.join(root, i))
	
	expResults = []
	for exp in experiment_dirs:
		flx_size = int(str(exp).split('_')[-2])
		flx_cat_dim = int(str(exp).split('_')[-1])
		expResult = ExpResultFull(flx_size, flx_cat_dim)
		expResult.add_all_possible_information_for_dir(exp)
		expResults.append(expResult)

	return expResults

def make_mnist_resutls_file_full(task_type_of_intersest, data = 'ex_7_mnist', all_information = False):
	print('\nRetrieving experiments\n')
	expResults = gather_results(data)
	data_type = 'emnist' if 'emnist' in data else 'mnist' if 'mnist' in data else 'fashion'
	if data_type != 'mnist':
		my_outfile = OUTFILE_all.replace('mnist', data_type)
	else:
		my_outfile = OUTFILE_all
	if all_information:
		my_outfile += '__all_information__'
	print(data_type)
	for exp in expResults:
		print(exp)


	expResultsSorted = sorted(expResults, key=lambda x: x.flx_size, reverse=False)

	with open(my_outfile.replace('_all_', '_{}_'.format(task_type_of_intersest)), 'w') as f:
		print(f)
		if task_type_of_intersest == 'classification':
			head = 'dataset, $|FL^A|$, $|FL^A_i|$, VAE-MSE, best ll, vtree method, compressed y, class acc'
		else:
			head = 'dataset, $|FL^{A,B}|$, $|FL^{A,B}_i|$, VAE-MSE, task type, best ll, vtree method, class acc'

		if all_information:
			head += ',cluster_id'
		f.write(head + '\n')

		for exp in expResultsSorted:
			for task_type in exp.exp_psdds:
				if not get_task_type_hiracy(task_type_of_intersest)(task_type):
					continue
				psdd_exp = exp.exp_psdds[task_type]
				for cluster_id in psdd_exp.classification_acc.keys():
					line = '{},\t{},\t{},\t{}'.format(data_type,exp.flx_size, exp.flx_cat_dim, exp.loss)
					# if task_type_of_intersest == 'classification':
					# 	line += ',\t2^{{{}}}'.format(exp.complexity_bin)
					if task_type_of_intersest != 'classification':
						line += ', \t{}'.format(task_type)
					line += ',\t{:.2},\t{},\t{:.4}'.format(psdd_exp.best_ll[cluster_id], \
							psdd_exp.vtree_method[cluster_id], max(psdd_exp.classification_acc[cluster_id]))
					if all_information:
						line += ',\t{}'.format(cluster_id)
					f.write(line + '\n')
	
def read_mnist_result_file(file):
	expResults = []
	names = []
	with open(file, 'r') as f:
		spamreader = csv.reader(f, delimiter=',')
		for idx,row in enumerate(spamreader):
			if idx == 0:
				names = row
				continue
			if len(row) < 4 or (row[0].strip() == 'emnist' or '#' in row[0]):
				continue

			flx_size = int(row[1].strip())
			flx_cat_dim = int(row[2].strip())
			loss = float(row[3].strip())
			expResult = ExpResult(flx_size, flx_cat_dim)
			expResult.add_loss(loss)
			expResults.append(expResult)
	return expResults




if __name__ == '__main__':
	# make_var_resutls_file()
	make_mnist_resutls_file_full('classification', data = 'ex_7_mnist')
	make_mnist_resutls_file_full('classification', data = 'ex_6_emnist')
	make_mnist_resutls_file_full('classification', data = 'ex_9_fashion')
	make_mnist_resutls_file_full('noisy',          data = 'ex_7_mnist')
	make_mnist_resutls_file_full('compositional',  data = 'ex_7_mnist')
	make_mnist_resutls_file_full('compositional',  data = 'ex_9_fashion')
import os,shutil


def clean_exp(experiment_name, cluster_id, min_it = 100):

	basedir = os.path.abspath(os.path.join('.', './experiments/{}/psdd_search_{}/learnpsdd_tmp_dir/'.format(experiment_name, cluster_id)))
	print(basedir)


	todel = []
	for root, dir_names, file_names in os.walk(os.path.join(basedir, './models/')):
		for file in file_names:
			iteration = int(file.split('_')[1])
			if iteration < min_it:
				todel.append(os.path.join(root, file))

	print(len(todel), ' files found')
	for file in todel:
		print('removing: ', file)
		os.remove(file)

def delete_all_but_best_k(experiment_name, cluster_id, best_k = 5):
	psdd_out_dir = os.path.abspath(os.path.join('.', './experiments/{}/psdd_search_{}/learnpsdd_tmp_dir/'.format(experiment_name, cluster_id)))
	_delete_all_but_best_k(psdd_out_dir, best_k = best_k)


def _delete_all_but_best_k(psdd_out_dir, best_k = 5):
	print('cleaning dir', psdd_out_dir)
	prgfile = os.path.abspath(os.path.join(psdd_out_dir, './progress.txt'))
	if not os.path.exists(prgfile) or os.path.getsize(prgfile) == 0:
		print('could not find  or empty prgfile for exp: {}'.format(psdd_out_dir))
		return

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
			last_it_ll = current_it
			it_ll[current_it] = current_ll
			it_weights[current_it] = []
			for idx, i in enumerate(range(4,num_learners + 4)):
				it_weights[current_it].append(float(splitted[i].strip()))

	if not it_ll or len(it_ll.keys()) <= best_k:
		print('no iterations could be found in file: {}'.format(psdd_out_dir))
		return

	best_k_it = []
	for i in range(best_k):
		best_k = max(it_ll, key=it_ll.get)
		best_k_it.append(best_k)
		print('best- {} = {}'.format(i, it_ll[best_k]))
		del it_ll[best_k]

	# print(best_k_it)

	todel = []
	for root, dir_names, file_names in os.walk(os.path.join(psdd_out_dir, './models/')):
		for file in file_names:
			iteration = int(file.split('_')[1])
			if iteration not in best_k_it:
				todel.append(os.path.join(root, file))

	print(len(todel), ' files found to delete')
	for file in todel:
		# print('removing: ', file)
		os.remove(file)

def free_most_space(best_k = 10):
	basedir = os.path.abspath(os.path.join('.', './experiments/'))
	toclean = []
	for root, dir_names, file_names in os.walk(basedir):
		for exp in dir_names:
			for root_2, dir_names_2, file_names_2 in os.walk(os.path.join(root, exp)):
				for psdd_seach_dir in dir_names_2:
					if 'psdd_search' in psdd_seach_dir:
						toclean.append(os.path.abspath(os.path.join(root_2, psdd_seach_dir, './learnpsdd_tmp_dir/')))

	for i in toclean:
		_delete_all_but_best_k(i, best_k = best_k)

if __name__ == '__main__':
	experiment_name = ''
	cluster_id = ''
	free_most_space()
	# delete_all_but_best_k('ex_7_mnist_16_4', 'james02')
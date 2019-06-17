import csv
import os

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
from scipy.interpolate import griddata

from .make_results_file import gather_only_var_results, gather_results, get_task_type_hiracy
from experiment import Experiment


RELATIVE_OUTFILE = './code/msc/output/var_results_mnist.csv'
file = os.path.abspath(os.path.join(os.environ['HOME'],RELATIVE_OUTFILE))


def make_heartmap():
	names = []
	fl_sizes = []
	cat_sizes = []
	losses  = []
	model_spaces = []
	model_spaces_values = {}

	with open(file, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',')
		for idx,row in enumerate(spamreader):
			if idx == 0:
				names = row
				continue
			if len(row) < 4 or (row[0].strip() == 'emnist' or '#' in row[0]):
				continue

			fl_sizes.append(int(row[1].strip()))
			cat_sizes.append(int(row[2].strip()))
			losses.append(float(row[3].strip()))
			model_idx = len(model_spaces)
			model_spaces.append(model_idx)
			model_spaces_values[model_idx] = row[4].strip()

	# create x-y points to be used in heatmap
	# xi = np.sort(np.unique(fl_sizes))
	yi = np.sort(np.unique(cat_sizes))
	xi = np.arange(np.min(fl_sizes), np.max(fl_sizes) + 1, 4)
	# print(xi, max(fl_sizes))
	# Z is a matrix of x-y values
	zi = griddata((fl_sizes, cat_sizes), losses, (xi[None,:], yi[:,None]), method='cubic')
	model_space = griddata((fl_sizes, cat_sizes), model_spaces, (xi[None,:], yi[:,None]), method='cubic')
	for i,t in enumerate(zi):
		for j,s in enumerate(t):
			if not xi[j] in fl_sizes:
				zi[i][j] = 'nan'
				model_space[i][j] = 'nan'

	# print('xi',xi)
	# print('yi',yi)
	for i in zi:
		print('zi ', i)
	for i in model_space:
		print('ms ', i)
	# Create the contour plot
	# CS = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow,
					  # vmax=max(losses), vmin=min(losses))
	fig_1 = plt.figure(figsize=(8, 4))
	ax = fig_1.add_subplot(111)

	extent = [xi[0], xi[-1], yi[0], yi[-1]]
	img = ax.imshow(zi)#, interpolation='nearest')#,extent=extent)
	plt.colorbar(img)
	# We want to show all ticks...
	ax.set_xticks(np.arange(len(xi)))
	ax.set_yticks(np.arange(len(yi)))
	# ... and label them with the respective list entries
	ax.set_xticklabels(xi)
	ax.set_yticklabels(yi)

	offsetx = 0.15
	offsety = -0.3

	for i in range(len(yi)):
		for j in range(len(xi)):
			idx = model_space[i, j]
			if str(idx) != 'nan':
				print(idx, i, j)
				idx = int(idx)
				# text = ax.text(j + offsetx, i + offsety, model_spaces_values[idx], ha="center", va="center", color="black")

	ax.grid(False)

	ax.set_title("mean squared (reconstuction) error for vae on mnist")
	ax.set_xlabel('# variables in the FL')
	ax.set_ylabel('categorical dim of FL variables')
	fig_1.savefig('./out/vae_results_mnist.pdf')
	plt.show()

def make_simple_graph_wrt_complexity():
	expResults = read_mnist_result_file()

	complexity_map = {}

	for exp in expResults:
		if exp.complexity_bin in complexity_map:
			complexity_map[exp.complexity_bin][exp.flx_cat_dim] = exp.loss
		else:
			complexity_map[exp.complexity_bin] = {exp.flx_cat_dim: exp.loss}

	print(complexity_map)
	x_axis = []
	y_axis = []

	for j in sorted(list(complexity_map.keys())):
		x_axis.append(j)
		y_axis.append(min(complexity_map[j].values()))

	print(x_axis)
	print(y_axis)

	fig_1 = plt.figure(figsize=(8, 4))
	ax = fig_1.add_subplot(111)
	ax.plot(x_axis, y_axis)

	# ax.grid(False)

	ax.set_title("mean squared (reconstuction) error for vae on mnist")
	ax.set_xlabel('#binary complexity of the feature layer (|model space| = 2^x)')
	ax.set_ylabel('mean squared (reconstuction) error')
	fig_1.savefig('./out/vae_results_mnist_bin_complexity.pdf')
	plt.show()

def make_vae_loss_graph_wrt_complexity(data = 'ex_7_mnist'):
	expResults = gather_only_var_results(data)

	data_type = 'emnist' if 'emnist' in data else 'mnist'

	complexity_map = {}

	for exp in expResults:
		if exp.complexity_bin in complexity_map:
			complexity_map[exp.complexity_bin][exp.loss] = [exp.flx_size, exp.flx_cat_dim]
		else:
			complexity_map[exp.complexity_bin] = {exp.loss: [exp.flx_size, exp.flx_cat_dim]}

	print(complexity_map)
	x_axis = []
	y_axis = []
	x_axis_best = []
	y_axis_best = []
	points = []

	for j in sorted(list(complexity_map.keys())):
		x_axis_best.append(j)
		best_loss = min(complexity_map[j].keys())
		y_axis_best.append(best_loss)
		for loss, identifier in complexity_map[j].items():
			x_axis.append(j)
			y_axis.append(loss)
			points.append(identifier)

	print(x_axis)
	print(y_axis)

	fig_1 = plt.figure(figsize=(20, 8))
	ax = fig_1.add_subplot(111)
	ax.plot(x_axis_best, y_axis_best)
	ax.scatter(x_axis, y_axis, marker = 'o')

	offsetx = 1
	offsety = 0.001
	for ii, v in enumerate(x_axis):
		# for jj in complexity_map[v].values():
		text = ax.text(v + offsetx, offsety + y_axis[ii], points[ii], ha="center", va="center", color="black")

	# ax.grid(False)

	ax.set_title("mean squared (reconstuction) error for vae on {}".format(data_type))
	ax.set_xlabel('#binary complexity of the feature layer (|model space| = 2^x)')
	ax.set_ylabel('mean squared (reconstuction) error')
	fig_1.savefig('./out/vae_results_{}_bin_complexity_annotated.pdf'.format(data_type))
	plt.show()

def make_acc_graph_wrt_complexity(data = 'ex_7_mnist', task_type_of_intersest = 'classification'):
	expResults = gather_results(data)

	data_type = 'emnist' if 'emnist' in data else 'mnist'

	complexity_map = {}

	for exp in expResults:
		for task_type in exp.exp_psdds:
			if not get_task_type_hiracy(task_type_of_intersest)(task_type):
				continue
			psdd_exp = exp.exp_psdds[task_type]
			for cluster_id in psdd_exp.classification_acc.keys():
				if exp.complexity_bin in complexity_map:
					complexity_map[exp.complexity_bin][max(psdd_exp.classification_acc[cluster_id])] = [exp.flx_size, exp.flx_cat_dim, psdd_exp.vtree_method[cluster_id],psdd_exp.compressed_y[cluster_id]]
				else:
					complexity_map[exp.complexity_bin] = {max(psdd_exp.classification_acc[cluster_id]): [exp.flx_size, exp.flx_cat_dim, psdd_exp.vtree_method[cluster_id],psdd_exp.compressed_y[cluster_id]]}

	print(complexity_map)
	x_axis = []
	y_axis = []
	x_axis_best = []
	y_axis_best = []
	points = []

	for j in sorted(list(complexity_map.keys())):
		x_axis_best.append(j)
		best_loss = max(complexity_map[j].keys())
		y_axis_best.append(best_loss)
		for loss, identifier in complexity_map[j].items():
			x_axis.append(j)
			y_axis.append(loss)
			points.append(identifier)

	print(x_axis)
	print(y_axis)

	fig_1 = plt.figure(figsize=(10, 6))
	ax = fig_1.add_subplot(111)
	ax.plot(x_axis_best, y_axis_best)
	ax.scatter(x_axis, y_axis, marker = 'o')

	offsetx = 1
	offsety = 0.001
	for ii, v in enumerate(x_axis):
		# for jj in complexity_map[v].values():
		text = ax.text(v + offsetx, offsety + y_axis[ii], points[ii], ha="center", va="center", color="black")

	# ax.grid(False)

	ax.set_title("classification acc for whole model on data: {}".format(data_type))
	ax.set_xlabel('#binary complexity of the feature layer (|model space| = 2^x)')
	ax.set_ylabel('classification acc on held out test set')
	fig_1.savefig('./plots/out/class_acc_{}_wrt_bin_complexity_annotated.pdf'.format(data_type))
	plt.show()

def make_acc_graph_wrt_ll(data = 'ex_7_mnist', task_type_of_intersest = 'classification'):
	expResults = gather_results(data)

	data_type = 'emnist' if 'emnist' in data else 'mnist'

	complexity_map = {}

	for exp in expResults:
		for task_type in exp.exp_psdds:
			if not get_task_type_hiracy(task_type_of_intersest)(task_type):
				continue
			psdd_exp = exp.exp_psdds[task_type]
			for cluster_id in psdd_exp.classification_acc.keys():
				x_axis_value = psdd_exp.best_ll[cluster_id] / float(exp.complexity_bin)
				if x_axis_value in complexity_map:
					complexity_map[x_axis_value][max(psdd_exp.classification_acc[cluster_id])] = [exp.flx_size, exp.flx_cat_dim, psdd_exp.vtree_method[cluster_id],psdd_exp.compressed_y[cluster_id]]
				else:
					complexity_map[x_axis_value] = {max(psdd_exp.classification_acc[cluster_id]): [exp.flx_size, exp.flx_cat_dim, psdd_exp.vtree_method[cluster_id],psdd_exp.compressed_y[cluster_id]]}

	print(complexity_map)
	x_axis = []
	y_axis = []
	x_axis_best = []
	y_axis_best = []
	points = []

	for j in sorted(list(complexity_map.keys())):
		x_axis_best.append(j)
		best_loss = max(complexity_map[j].keys())
		y_axis_best.append(best_loss)
		for loss, identifier in complexity_map[j].items():
			x_axis.append(j)
			y_axis.append(loss)
			points.append(identifier)

	print('x_axis',x_axis)
	print('y_axis',y_axis)

	fig_1 = plt.figure(figsize=(20, 8))
	ax = fig_1.add_subplot(111)
	ax.plot(x_axis_best, y_axis_best)
	ax.scatter(x_axis, y_axis, marker = 'o')

	offsetx = np.diff(ax.get_xlim())/50.0 #1
	offsety = np.diff(ax.get_ylim())/50.0#0.001
	for ii, v in enumerate(x_axis):
		# for jj in complexity_map[v].values():
		text = ax.text(v + offsetx, offsety + y_axis[ii], points[ii], ha="center", va="center", color="black")

	# ax.grid(False)

	ax.set_title("classification acc for whole model on data: {}".format(data_type))
	ax.set_xlabel('#binary complexity of the feature layer (|model space| = 2^x) over the ll [|model space|/ll]')
	ax.set_ylabel('classification acc on held out test set')
	fig_1.savefig('./out/class_acc_{}_wrt_ll_annotated.pdf'.format(data_type))
	plt.show()

def make_acc_graph_wrt_noisy_level(data = 'ex_7_mnist', task_type_of_intersest = 'noisy'):
	expResults = gather_results(data)

	data_type = 'emnist' if 'emnist' in data else 'mnist'

	complexity_map = {}

	for exp in expResults:
		for task_type in exp.exp_psdds:
			if not get_task_type_hiracy(task_type_of_intersest)(task_type):
				continue
			psdd_exp = exp.exp_psdds[task_type]
			for cluster_id in psdd_exp.classification_acc.keys():
				x_axis_value = int(task_type.split('-')[1])
				if x_axis_value in complexity_map:
					complexity_map[x_axis_value][max(psdd_exp.classification_acc[cluster_id])] = [exp.flx_size, exp.flx_cat_dim, psdd_exp.vtree_method[cluster_id],psdd_exp.compressed_y[cluster_id]]
				else:
					complexity_map[x_axis_value] = {max(psdd_exp.classification_acc[cluster_id]): [exp.flx_size, exp.flx_cat_dim, psdd_exp.vtree_method[cluster_id],psdd_exp.compressed_y[cluster_id]]}

	print(complexity_map)
	x_axis = []
	y_axis = []
	x_axis_best = []
	y_axis_best = []
	points = []

	for j in sorted(list(complexity_map.keys())):
		x_axis_best.append(j)
		best_loss = max(complexity_map[j].keys())
		y_axis_best.append(best_loss)
		for loss, identifier in complexity_map[j].items():
			x_axis.append(j)
			y_axis.append(loss)
			points.append(identifier)

	print('x_axis',x_axis)
	print('y_axis',y_axis)

	fig_1 = plt.figure(figsize=(10, 6))
	ax = fig_1.add_subplot(111)
	ax.plot(x_axis_best, y_axis_best)
	ax.scatter(x_axis, y_axis, marker = 'o')

	offsetx = np.diff(ax.get_xlim())/50.0 #1
	offsety = np.diff(ax.get_ylim())/50.0#0.001
	for ii, v in enumerate(x_axis):
		# for jj in complexity_map[v].values():
		text = ax.text(v + offsetx, offsety + y_axis[ii], points[ii], ha="center", va="center", color="black")

	# ax.grid(False)

	ax.set_title("classification acc for whole model on data: {}".format(data_type))
	ax.set_xlabel('noisy level (how many noisy labels where added while training)')
	ax.set_ylabel('classification acc on held out test set')
	fig_1.savefig('./plots/out/class_acc_{}_wrt_noisy_level.pdf'.format(data_type))
	plt.show()

def plot_psdd_learning(exp):
	prgfile = os.path.abspath(os.path.join(exp.psdd_out_dir, './learnpsdd_tmp_dir/progress.txt'))
	if not os.path.exists(prgfile):
		return

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

	x_axis = it_ll.keys()
	y_axis = it_ll.values()

	fig_1 = plt.figure(figsize=(8, 4))
	ax = fig_1.add_subplot(111)
	ax.plot(x_axis, y_axis)

	ax.set_title("Psdd learning for: {}".format(exp.identifier()))
	ax.set_xlabel('#iterations')
	ax.set_ylabel('ll wrt. size of tree')
	outpath = os.path.join(exp.evaluation_dir_path, './psdd_learning.pdf')
	fig_1.savefig(outpath)
	print('saving fig to {}'.format(outpath))

def plot_all_psdd_learning():
	BLOODBORN_BASE = os.path.abspath(os.path.join(os.environ['HOME'],'./code/msc/output/experiments/'))

	toclassify = []
	for root, dir_names, file_names in os.walk(BLOODBORN_BASE):
		for experiment_parent_name in dir_names:
			for experiment_dir_path, dir_names_2, file_names_2 in os.walk(os.path.join(root, experiment_parent_name)):
				for psdd_search_dir in dir_names_2:
					for possible_names in ['psdd_search_', 'psdd_model_']:
						if possible_names in psdd_search_dir:
							identifier = str(psdd_search_dir).split(possible_names)[1].replace('/','')
							if len(identifier.split('_')) == 3 or (len(identifier.split('_')) == 2 and 'james' in identifier):
								cluster_id = '_'.join(identifier.split('_')[:-1])
								task_type = identifier.split('_')[-1]
							else:
								cluster_id = identifier
								task_type = 'classification'

							exp = Experiment(experiment_parent_name, cluster_id, task_type)
							
							if exp is None:
								continue

							progressfile = os.path.join(exp.psdd_out_dir, './learnpsdd_tmp_dir/progress.txt')
							if not os.path.exists(progressfile) or os.path.getsize(progressfile) == 0:
								continue

							if not os.path.exists(exp.evaluation_dir_path):
								continue
								# print('added because evaldir does not exist', evaluationDir)
							elif exp.fl_info is not None:
								toclassify.append(exp)
	for i in toclassify:
		print(i)
		plot_psdd_learning(i)

if __name__ == '__main__':

	# make_vae_loss_graph_wrt_complexity('ex_7_mnist')
	# make_vae_loss_graph_wrt_complexity('ex_6_emnist')
	# make_acc_graph_wrt_complexity('ex_7_mnist')
	# make_acc_graph_wrt_ll('ex_7_mnist')
	make_acc_graph_wrt_noisy_level()
	# plot_all_psdd_learning()











import csv
import os

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
from scipy.interpolate import griddata

from make_results_file import read_mnist_result_file


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
	fig_1.savefig('./vae_results_mnist.pdf')
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
	fig_1.savefig('./vae_results_mnist_bin_complexity.pdf')
	plt.show()

def make_graph_wrt_complexity():
	expResults = read_mnist_result_file()

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

	ax.set_title("mean squared (reconstuction) error for vae on mnist")
	ax.set_xlabel('#binary complexity of the feature layer (|model space| = 2^x)')
	ax.set_ylabel('mean squared (reconstuction) error')
	fig_1.savefig('./vae_results_mnist_bin_complexity_annotated.pdf')
	plt.show()


if __name__ == '__main__':

	make_graph_wrt_complexity()














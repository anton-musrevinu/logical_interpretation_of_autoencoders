import os
import numpy as np
import csv


RELATIVE_BASEDIR = './code/msc/output/experiments/'
RELATIVE_OUTFILE = './code/msc/output/var_results_mnist.csv'
BASEDIR = os.path.abspath(os.path.join(os.environ['HOME'],RELATIVE_BASEDIR))
OUTFILE = os.path.abspath(os.path.join(os.environ['HOME'],RELATIVE_OUTFILE))

class ExpResult(object):
	def __init__(self,flx_size, flx_cat_dim):
		self.flx_size = flx_size
		self.flx_cat_dim = flx_cat_dim
		self.complexity_num = flx_cat_dim ** flx_size
		self.complexity_bin = int(np.ceil(np.log2(self.flx_cat_dim))) * self.flx_size

	def add_loss(self,loss):
		self.loss = loss

	def add_losses_from_dir(self,dir):
		summary = os.path.abspath(os.path.join(dir, './VAEManager/summary_training.txt'))
		with open(summary, 'r') as f:
			for_right_error = False
			for line in f:
				if line.startswith('for error: valid_MSE'):
					for_right_error = True
				if line.strip().startswith('valid_MSE:'):
					loss = float(line.strip().split(':')[-1])
		self.loss = loss

		print('bin complexity: {} - loss found: {}'.format(self.complexity_bin,self.loss))

def make_mnist_resutls_file():
	experiment_dirs = []
	for root, dir_names, file_names in os.walk(BASEDIR):
		for i in dir_names:
			if 'ex_7_mnist' in i:
				experiment_dirs.append(os.path.join(root, i))
	
	expResults = []
	for exp in experiment_dirs:
		flx_size = int(str(exp).split('_')[-2])
		flx_cat_dim = int(str(exp).split('_')[-1])
		expResult = ExpResult(flx_size, flx_cat_dim)
		expResult.add_losses_from_dir(exp)
		expResults.append(expResult)

	with open(OUTFILE, 'w') as f:
		f.write('dataset, FL categorical size, categorical dimension, MSE, model space complexity\n')
		for exp in expResults:
			line = 'mnist,\t{},\t{},\t{},\t2^{{{}}}'.format(exp.flx_size, exp.flx_cat_dim, exp.loss, exp.complexity_bin)
			f.write(line + '\n')
	
def read_mnist_result_file():
	expResults = []
	names = []
	with open(OUTFILE, 'r') as f:
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
	make_mnist_resutls_file()
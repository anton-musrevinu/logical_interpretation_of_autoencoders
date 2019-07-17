import numpy as np
import os

class FlDomainInfo(object):

	def __init__(self, name, nb_vars, var_cat_dim, binary_encoded,encoded_start_idx,encoded_end_idx):
		self.name = name
		self.nb_vars = int(nb_vars)         		#number of variables
		self.var_cat_dim = int(var_cat_dim) 		#variable_categorical_dim
		self.bin_encoded = int(binary_encoded) 	#are the variables binary encoded
		self.encoded_start_idx = int(encoded_start_idx)
		self.encoded_end_idx = int(encoded_end_idx)

	def __str__(self):
		return '{},{},{},{},{},{}'.format(self.name, self.nb_vars, self.var_cat_dim, self.bin_encoded, self.encoded_start_idx, self.encoded_end_idx)

	def get_values_as_str_list(self):
		return '{},{},{},{},{}'.format(self.nb_vars, self.var_cat_dim, self.bin_encoded, self.encoded_start_idx, self.encoded_end_idx)

	def get_empty_example(self):
		if self.bin_encoded:
			return np.zeros((self.nb_vars, self.var_cat_dim))
		else:
			return np.zeros(self.encoded_end_idx - self.encoded_start_idx)

def write_fl_batch_to_file(file_encoded_path, flx_categorical, fly_onehot, batch_idx, compress_fly = True):
	flx_binary = encode_flx_to_binary_batch(flx_categorical)
	if compress_fly:
		fly = encode_onehot_to_binary_batch(fly_onehot)
	else:
		fly = fly_onehot

	fl = np.concatenate((flx_binary, fly), axis = 1)

	if batch_idx == 0:
		edit_str = 'w'
		flx_info = FlDomainInfo('flx', flx_categorical.shape[1], flx_categorical.shape[2], True, 0, flx_binary.shape[1])
		fly_info = FlDomainInfo('fly', 1, fly_onehot.shape[1], compress_fly, flx_binary.shape[1], fl.shape[1])
		fl_info = [flx_info, fly_info]
		create_info_file(file_encoded_path, fl_info)
		# print('[encode] - creating ennoded dataset with flx: {}, fly: {}, fl: {}'.format(flx_binary.shape, fly.shape, fl.shape))
	else:
		edit_str = 'a'

	with open(file_encoded_path,edit_str) as f:
		for row in fl:
			# print(row)
			row_str = ''.join(['%.0f,' % num for num in row])
			row_str = row_str[:-1] + '\n'
			# row_str = row_str + '\n'
			f.write(row_str)

	return fl.shape[1]

def write_fl_batch_to_file_new(file_encoded_path, fls_data, fl_info, batch_idx):
	fls_encoded = []
	for idx, fl_domain in fls_data.items():
		if fl_info[idx].bin_encoded and fl_info[idx].name == 'fly':
			fls_encoded.append(encode_onehot_to_binary_batch(fl_domain))
		elif fl_info[idx].bin_encoded and not fl_info[idx].name == 'fly':
			fls_encoded.append(encode_flx_to_binary_batch(fl_domain))
		elif fl_info[idx].name == 'fly':
			fls_encoded.append(fl_domain)
		else:
			columns = []
			#iterate over the individual variables
			for i in range(fl_domain.shape[1]):
				column = fl_domain[:,i,:]
				columns.append(column)

			fl_domain = np.concatenate(columns, axis = 1)
			fls_encoded.append(fl_domain)
		# print(fls_encoded[-1])

	fl_all = np.concatenate(fls_encoded, axis = 1)

	if batch_idx == 0:
		edit_str = 'w'
		create_info_file(file_encoded_path, fl_info)
	else:
		edit_str = 'a'

	with open(file_encoded_path,edit_str) as f:
		for row in fl_all:
			# print(row)
			row_str = ''.join(['%.0f,' % num for num in row])
			row_str = row_str[:-1] + '\n'
			# row_str = row_str + '\n'
			f.write(row_str)

	return fl_all.shape[1]


def create_info_file(file_encoded_path, fl_info):
	with open(file_encoded_path + '.info', 'w') as f:
		f.write('domain name; nb_vars, var_cat_dim, binary_encoded,encoded_start_idx,encoded_end_idx\n')
		for fl_domain_info in fl_info:
			f.write('{}\n'.format(fl_info[fl_domain_info]))

def read_info_file_basic(info_file):
	domains = {}
	with open(info_file, 'r') as f:
		for line_idx, line in enumerate(f):
			if line_idx == 0 or len(line.strip().split(',')) < 5 or line.startswith('encoded_data_dir'):
				continue
			# print(line, line.replace('\n','').split(','))
			fl_info = FlDomainInfo(*line.split(','))
			domains[fl_info.name] = fl_info
	return domains

	# print('[INFO] \t\t\t- fl_info read {} from file: {}'.format(domains, '/'.join(file_encoded_path.split('/')[-3:])))
	# print('[INFO] \t\t\t info file read succesf

def read_info_file(file_encoded_path):
	return read_info_file_basic(file_encoded_path + '.info')

def recreate_fl_info_for_old_experiments(exeriment_dir):
	exeriment_dir = os.path.abspath(exeriment_dir)
	domains = {}
	experiment_name = exeriment_dir.split('/')[-2]
	print('recreate_fl_info_for_old_experiments', 'experiment_name',experiment_name, exeriment_dir)

	flx_nb_vars = int(experiment_name.split('_')[3])
	flx_var_cat_dim = int(experiment_name.split('_')[4])
	flx_bin_encoded = 1
	flx_encoded_start_idx = 0
	flx_encoded_end_idx = int(np.ceil(np.log2(flx_var_cat_dim))) * flx_nb_vars
	domains['flx'] = FlDomainInfo('flx', flx_nb_vars, flx_var_cat_dim, flx_bin_encoded, flx_encoded_start_idx, flx_encoded_end_idx)

	fly_nb_vars = 1
	fly_var_cat_dim = 47 if 'emnist' in experiment_name else 10
	fly_bin_encoded = 1
	fly_encoded_start_idx = flx_encoded_end_idx
	fly_encoded_end_idx = 6 + fly_encoded_start_idx if 'data_bug' in experiment_name else int(np.ceil(np.log2(fly_var_cat_dim))) + fly_encoded_start_idx
	domains['fly'] = FlDomainInfo('fly', fly_nb_vars, fly_var_cat_dim, fly_bin_encoded, fly_encoded_start_idx, fly_encoded_end_idx)

	return domains


def encode_flx_to_binary_batch(flx_categorical):
	flx_binary_size = int(np.ceil(np.log2(flx_categorical.shape[2])))
	vec_func = np.vectorize(convert_onehot_to_binary,otypes=[np.ndarray], signature = '(m),()->(t)')
	columns = []
	#iterate over the individual variables
	for i in range(flx_categorical.shape[1]):
		column = flx_categorical[:,i,:]
		column_as_bin = vec_func(column, flx_binary_size)
		columns.append(column_as_bin)

	flx_binary = np.concatenate(columns, axis = 1)
	return flx_binary

def encode_onehot_to_binary_batch(onehot_batch):
	fly_binary_size = int(np.ceil(np.log2(onehot_batch.shape[1])))

	vec_func = np.vectorize(convert_onehot_to_binary,otypes=[np.ndarray], signature = '(m),()->(t)')
	labels_as_bin = vec_func(onehot_batch, fly_binary_size)

	# print(fly_binary_size)

	# print(labels_as_bin.shape, type(labels_as_bin))
	# for i in range(onehot_batch.shape[0]):
	# 	print(onehot_batch[i],labels_as_bin[i])
		# for idx_label, line in onehot_batch:
		# 	cat_value_bin = np.binary_repr(torch.argmax(line),fly_binary_size)
		# 	binary_list = torch.IntTensor(list(map(int,cat_value_bin)))
		# 	labels_as_bin[idx_label] = binary_list
	return labels_as_bin

def convert_onehot_to_binary(onehot_vec, fl_binary_size):
	cat_value_bin = np.binary_repr(np.argmax(onehot_vec),fl_binary_size)
	binary_vec = np.asarray(list(map(int,cat_value_bin)))
	return binary_vec


def decode_binary_to_onehot(binary_list, cat_dim):
	value_as_int = decode_binary_to_int(binary_list)
	value_as_onehot = np.zeros((cat_dim))
	value_as_onehot[value_as_int] = 1
	return value_as_onehot

def decode_binary_to_int(binary_list):
	value_as_bin = ''.join(binary_list).replace('\n','')
	value_as_int = int(value_as_bin,2)
	# print(binary_list, value_as_bin, value_as_int)
	return value_as_int
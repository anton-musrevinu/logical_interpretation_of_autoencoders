import numpy as np

def decode_binary_to_onehot(binary_list, cat_dim):
	value_as_int = decode_binary_to_int(binary_list)
	value_as_onehot = np.zeros((cat_dim))
	value_as_onehot[value_as_int] = 1
	return value_as_onehot

def decode_binary_to_int( binary_list):
	value_as_bin = ''.join(binary_list).replace('\n','')
	value_as_int = int(value_as_bin,2)
	print(binary_list, value_as_bin, value_as_int)
	return value_as_int
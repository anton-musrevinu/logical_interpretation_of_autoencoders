import numpy as np

def write_batch_to_file(file,data_batch,add = True, first_line = None):
	if add:
		edit_str = 'a'
	else:
		edit_str = 'w'
		# print(data_batch)
	with open('{}'.format(file),edit_str) as f:
		data_batch = data_batch.detach().data.numpy()
		# if not add and first_line != None:
		# 	f.write('{}\n'.format(first_line))
		for row in data_batch:
			# print(row)
			row_str = ''.join(['%.0f,' % num for num in row])
			row_str = row_str[:-1] + '\n'
			# row_str = row_str + '\n'
			f.write(row_str)
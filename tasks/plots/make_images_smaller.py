import os,sys

from PIL import Image
import PIL

import numpy as np

def do_make_class_samples_smaller_and_sep(dir_to_query, image_size = 28, new_nb_rows = 3):
	print('searching ', dir_to_query)
	rowtotake = 5
	nb_columns = 5
	for root, folders, files in os.walk(dir_to_query):
		for file in files:
			if file.endswith('.png'): #and not 'small' in file:
				print('found file: ', file)
				image = Image.open(os.path.join(root,file))
				padding = int(infer_offest(image.size[1], image_size, 1) / 2)
				rows = (image.size[1] - (padding * 2) ) / (image_size + (padding * 2))

				# new_width = image.size[0]
				new_width = (nb_columns * 2) * (image_size + (padding)) + padding * 2

				box = (0, rowtotake * (image_size + (padding * 2)), new_width, (new_nb_rows + rowtotake) * (image_size + (padding * 2)) + padding * 2)    
				small_image = image.crop(box)
				small_image.save(os.path.join(root, file.replace('.png','_small.png')))

def add_buffer_to_all_images(dir_to_query, image_size = 28):
	for root, folders, files in os.walk(dir_to_query):
		for file in files:
			if file.endswith('.png') and not 'buffer' in file and not file.startswith('._'):
				print('starting on file: {}'.format(file))
				image = Image.open(os.path.abspath(os.path.join(root,file)))
				padding = int(infer_offest(image.size[1], image_size, 1) / 2)
				nb_rows = int((image.size[1] - (padding * 2) ) / (image_size + (padding * 2)))
				nb_columns = int(((image.size[0] - (padding * 2) ) / (image_size + (padding))) /2)
				print(nb_rows, nb_columns)
				new_image = add_buffer(image, padding, nb_rows, nb_columns,image_size)
				new_image.save(os.path.join(root, file.replace('.png', '_buffer.png')))


def add_buffer(image, padding, nb_rows,nb_columns, image_size):
	np_image = np.array(image) 
	np_image = np.mean(np.array(np_image), axis = 2)
	new_columns = []
	pair_width = (image_size + padding) * 2
	for i in range(nb_columns):
		starting_width = (pair_width) * (i) + padding
		end_width = (pair_width) * (i + 1) + padding * 2
		np_column = np_image[:,starting_width:end_width]
		print('column\t\t',np_column.shape)
		new_columns.append(np_column)
		if i < nb_columns -1:
			buffer_column = build_buffer(image_size, nb_rows, padding)
			print('buffer_column\t', buffer_column.shape)
			new_columns.append(buffer_column)
		if i == 0:
			print(np.max(np_column),np.min(np_column))
			print(np.max(buffer_column),np.min(buffer_column))

	new_np_image = np.concatenate((new_columns), axis = 1)
	new_image = Image.fromarray(np.uint8(new_np_image), 'L')
	return new_image


def build_buffer(image_size, nb_rows,padding):
	buffers = []
	buffer_width = 2
	for i in range(nb_rows):
		top_bottom = np.zeros((int(padding/2),buffer_width))
		middle = np.ones((image_size + padding, buffer_width))
		buffers.append(np.concatenate((top_bottom, middle, top_bottom), axis =0))
	column_buffer = np.concatenate((buffers), axis = 0)
	top_bottom = np.zeros((padding,buffer_width))
	column_buffer_real = np.concatenate((top_bottom, column_buffer, top_bottom), axis = 0)
	left_right = np.zeros((column_buffer_real.shape[0], padding * 3))
	column_buffer_real = np.concatenate((left_right, column_buffer_real, left_right), axis = 1)
	return column_buffer_real * 255
		# buffers.append(np.concatenate((np.zeros((diff_array_x.shape[0], 2)),np.ones((diff_array_x.shape[0], padding_between_images)), np.zeros((diff_array_x.shape[0], 2))), axis = 1)


def infer_offest(panel_image_size, image_size, current_offset):
	remainder = float(panel_image_size - current_offset) % (image_size + current_offset)
	if remainder == 0:
		# print('[INFO] -- offset found with value: {}'.format(current_offset))
		return current_offset
	else:
		return infer_offest(panel_image_size, image_size, current_offset + 1)


if __name__ == '__main__':
	dir_to_query = os.path.abspath(os.path.join(os.environ['HOME'], './University/thesis_msc/publish/AAAI%20Papaer%20-%20Symbolic%20Interpretations%20of%20Autoencoders 2/figures/relational/'))

	add_buffer_to_all_images(dir_to_query)





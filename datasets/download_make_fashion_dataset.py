import os, shutil
import gzip
import numpy as np

def load_mnist(path, kind='train'):
	"""Load MNIST data from `path`"""
	labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz' % kind)
	images_path = os.path.join(path,'%s-images-idx3-ubyte.gz' % kind)

	with gzip.open(labels_path, 'rb') as lbpath:
		labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

	with gzip.open(images_path, 'rb') as imgpath:
		images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

	return images, labels

def download_split_fashion():
	os.mkdir('./tmp')
	os.system('git clone https://github.com/zalandoresearch/fashion-mnist.git tmp')
	
	for i in ['train', 't10k']:
		inputs, labels = load_mnist('./tmp/data/fashion/', i)
		print('set: {} - data: {}'.format(i, inputs.shape))
		if i == 'train':
			inputs_train = inputs[:50000]
			inputs_val = inputs[50000:]
			targets_train = labels[:50000]
			targets_val = labels[50000:]

			np.savez('./fashion-train.npz', inputs = inputs_train, targets = targets_train)
			np.savez('./fashion-valid.npz', inputs = inputs_val, targets = targets_val)
		else:
			np.savez('./fashion-test.npz', inputs = inputs, targets = labels)

	print('\nfashion dataset: ')
	for i in ['train', 'test', 'valid']:
		loaded = np.load('./fashion-{}.npz'.format(i))
		inputs = loaded['inputs'].astype(np.float32)
		print('set: {} - data: {}'.format(i, inputs.shape))

	shutil.rmtree('./tmp')

def make_example_dataset(percent = 0.001):
	for i in ['train', 'valid', 'test']:
		loaded = np.load('./mnist-{}.npz'.format(i))
		inputs = loaded['inputs'].astype(np.float32)
		labels = loaded['targets'].astype(np.float32)
		inputs_example = inputs[:int(percent * len(inputs))]
		labels_example = labels[:int(percent * len(labels))]
		print('set: {} - inputs_example: {}, labels_example: {}'.format(i,inputs_example.shape, labels_example.shape))
		np.savez('./mnist-example-{}.npz'.format(i), inputs = inputs_example, targets = labels_example)

if __name__ == '__main__':
	# download_split_fashion()
	make_example_dataset()











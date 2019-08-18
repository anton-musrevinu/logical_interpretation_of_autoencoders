from __future__ import print_function

import sys
import os, sys, tarfile, errno
import numpy as np
import matplotlib.pyplot as plt
    
if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib # ugly but works
else:
    import urllib

try:
    from imageio import imsave
except:
    from scipy.misc import imsave

print(sys.version_info) 

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

import os,sys
DATADIR = os.path.dirname(os.path.realpath(__file__))
# path to the directory with the data
DATA_DIR = os.path.abspath(os.path.join(DATADIR,'./tmp_sln'))

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
DATA_PATH = os.path.abspath(os.path.join(DATADIR,'./tmp_sln/stl10_binary/train_X.bin'))

# path to the binary train file with labels
LABEL_PATH = os.path.abspath(os.path.join(DATADIR,'./tmp_sln/stl10_binary/train_y.bin'))

UNLABELED_DATA_PATH = os.path.abspath(os.path.join(DATADIR,'./tmp_sln/stl10_binary/unlabeled_X.bin'))

SAVE_DIR = DATADIR

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    # image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()

def save_image(image, name):
    imsave("%s.png" % name, image, format="png")

def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def save_images(images, labels):
    print("Saving images to disk")
    i = 0
    for image in images:
        label = labels[i]
        directory = './img/' + str(label) + '/'
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = directory + str(i)
        print(filename)
        save_image(image, filename)
        i = i+1

def save_images_unlabeled(path_to_data):
    TRAIN_PER = .9

    inputs = read_all_images(path_to_data)
    print(inputs.shape)
    inputs_train = inputs[:int(TRAIN_PER * inputs.shape[0])]
    inputs_valid = inputs[int(TRAIN_PER * inputs.shape[0]):]
    # np.savez(os.path.join(DATADIR, './sln-unsupervised-train.npz'), inputs = inputs_train)
    # np.savez(os.path.join(DATADIR, './sln-unsupervised-valid.npz'), inputs = inputs_valid)
    if not os.path.exists(os.path.join(DATADIR, './sln-unsupervised')):
        os.mkdir(os.path.join(DATADIR, './sln-unsupervised'))
    train_dir = os.path.join(DATADIR, './sln-unsupervised/train')
    valid_dir = os.path.join(DATADIR, './sln-unsupervised/valid')

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)

    for idx, image in enumerate(inputs_train):
        save_image(image, os.path.join(train_dir, './{}'.format(idx)))

    for idx, image in enumerate(inputs_valid):
        save_image(image, os.path.join(valid_dir, './{}'.format(idx)))



def save_images_numpy(images, labels):
    #Unlabbeled (100000) data split:
    #train: 90%
    #valid: 10%

    #Labeled data:
    #psdd.train: 500 images x 80% = 400
    #psdd.valid: 50 images
    #psdd.test:  50 images
    #test: 800 (x 10 ) test images per class (should be left untouched)

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
    
if __name__ == "__main__":
    # download data if needed
    download_and_extract()

    # # test to check if the image is read correctly
    # with open(DATA_PATH) as f:
    #     image = read_single_image(f)
    #     plot_image(image)

    # # test to check if the whole dataset is read correctly
    # images = read_all_images(DATA_PATH)
    # print(images.shape)

    # labels = read_labels(LABEL_PATH)
    # print(labels.shape)

    save_images_unlabeled(UNLABELED_DATA_PATH)

    # save images to disk
    # save_images(images, labels)
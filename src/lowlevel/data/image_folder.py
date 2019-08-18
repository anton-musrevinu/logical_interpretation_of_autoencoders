"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
import numpy as np
import torch

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def default_loader(path):
    return Image.open(path)#.convert('RGB')

class ImageDataset(BaseDataset):
    def __init__(self, opt, domain, type_of_data, mydir = None):
        BaseDataset.__init__(self, opt)

        transform_list = []
        # transform_list.append(transforms.Resize((opt.image_width, opt.image_height), Image.BICUBIC))
        transform_list.append(lambda img: transforms.functional.rotate(img, 90))
        if opt.input_nc == 1:
            transform_list.append(transforms.Grayscale(1))
        transform = transforms.Compose(transform_list)

        self.image_width = opt.image_width
        self.image_height = opt.image_height

        dir = os.path.join(os.path.abspath(opt.dataroot), './{}/'.format(domain))
        self.dataset = ImageFolder(dir, opt.num_classes, transform=transform, testing_data = opt.testing)
        # print('self.dataset created: ', self.dataset)
        self.num_classes = opt.num_classes
        # print('finished init base dataset: ', self)

    def __len__(self):
        # size = int(float(len(self.dataset))/self.opt.batch_size) * self.opt.batch_size
        return int(float(len(self.dataset))/self.opt.batch_size) * self.opt.batch_size

    def __str__(self):
        return 'dataset: {}, length: {}'.format(str(self.dataset), len(self))

    def get_input_shape(self):
        return self.dataset.get_input_shape()

    def __getitem__(self, index):
        # print('-- passing along images for index ', index)
        t = self.dataset[index]
        # print('returning {}, {} for index: {}'.format(type(t), t, index))
        return t

class ImageFolder(data.Dataset):

    def __init__(self, root, num_classes, transform=None, testing_data = False, loader = default_loader):
        imgs = make_dataset(root, 20 if testing_data else -1)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        # print(len(imgs))
        self.num_classes = num_classes
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.inputs_shape = self[0]['inputs'].shape
        # print(self)

    def __str__(self):
        return 'shape: {}x{}'.format(len(self.imgs), self.inputs_shape)#, self.num_classes)

    def __getitem__(self, index):
        path = self.imgs[index % len(self.imgs)]
        # print('hi - {}'.format(path))
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        np_image = np.array(img)#.reshape(image_size[0], image_size[1])
        if len(np_image.shape) == 2:
            np_image = np_image.reshape(1, np_image.shape[0], np_image.shape[1])
        else:
            np_image = np.transpose(np_image, (2, 1, 0))

        # print(index, self.targets[index], type(index), type(self.targets[index]))
        # target_elem = self.to_one_of_k(int(label))
        if np.max(np_image) > 1:
            np_image = np_image/255
        input_elem = torch.Tensor(np_image).float()
        np_image = None
        # target_elem = torch.Tensor(target_elem).float()
        # print(type(input_elem), type(target_elem))
        # print('--- returning image_elem: {} for index: {}'.format(input_elem.shape, index))

        return {'inputs': input_elem, 'indexs': index}

        # if self.return_paths:
        #     return img, path
        # else:
        #     return img

    def __len__(self):
        return len(self.imgs)

# def default_loader(path):
#     return Image.open(path).convert('RGB')


# class ImageFolder_old(data.Dataset):

#     def __init__(self, root, transform=None, return_paths=False,
#                  loader=default_loader):
#         imgs = make_dataset(root)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in: " + root + "\n"
#                                "Supported image extensions are: " +
#                                ",".join(IMG_EXTENSIONS)))

#         self.root = root
#         self.imgs = imgs
#         self.transform = transform
#         self.return_paths = return_paths
#         self.loader = loader

#     def __getitem__(self, index):
#         path = self.imgs[index]
#         img = self.loader(path)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.return_paths:
#             return img, path
#         else:
#             return img

#     def __len__(self):
#         return len(self.imgs)

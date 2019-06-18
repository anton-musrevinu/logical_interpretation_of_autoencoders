import numpy as np
from os import walk
import os
import sys
from PIL import Image
import PIL
from torchvision.utils import save_image,make_grid
import random

IMG_EXTENSIONS = [
    '.png'
]

def read_expiriment_data(expirimentDir):
    total_losses = {}
    indexes = {}
    with open('{}/CycleGanManager/result_outputs/summary.txt'.format(expirimentDir),'r') as f:
        for lineNumber, line in enumerate(f):
            if lineNumber == 0:
                for idx,elem in enumerate(line.split(',')):
                    total_losses[elem] = []
                    indexes[idx] = elem

            else:
                for idx, elem in enumerate(line.split(',')):
                    elem = elem.replace('/n','')
                    total_losses[indexes[idx]].append(float(elem))

    for key in total_losses.keys():
        total_losses[key] = np.array(total_losses[key])
        print(total_losses[key])

    return total_losses

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def infer_offest(panel_image_size, image_size, current_offset):
    remainder = float(panel_image_size - current_offset) % (image_size + current_offset)
    if remainder == 0:
        return current_offset
    else:
        return infer_offest(panel_image_size, image_size, current_offset + 1)


def get_examples_image_for_epoch(save_dir, epoch_idx, image_size = 28):
    path = os.path.join(save_dir, 'transfer_example_epoch_{}.png'.format(epoch_idx))
    image = Image.open(path)
    elems = 21
    offset = infer_offest(image.size[1], image_size, 1)
    rows = (image.size[1] - offset ) / (image_size + offset)
    # print('number of rows: {}'.format(rows))
    columns = 3
    whole_image_w = (image.size[0] - offset ) / columns
    # print(rows, image_size, epoch_idx, image, columns, whole_image_w)
    line_num = random.randint(0, rows - 1)
    # padding = int((image.size[0] - 6 * image_size) / 7)
    padding = 2
    box = (0, line_num * (image_size + padding) + padding, whole_image_w + 2 * padding, (line_num + 1) * (image_size + padding))    
    image_line = image.crop(box)#left, upper, right, lower

    # print(box,image.size, image_line.size)
    # image_line.show()
    return image_line
    # image_line.show( )

def make_time_image(image_dir, image_size = 28):
    # name = dir.split('/')[-1]
    # image_dir = os.path.join(dir,'./CycleGanManager/result_outputs/')
    file_names = []
    for root, dir_names, file_names_ in sorted(os.walk(image_dir)):
        for i in file_names_:
            if is_image_file(i) and 'transfer_example_epoch' in i:
                file_names.append(i)
    # print(file_names)
    # print(len(file_names))

    num_of_examples_to_show = min(20, len(file_names))

    padding = 2
    # print('num_of_examples_to_show',num_of_examples_to_show)

    save_epochs = list(map(int, np.linspace(1,len(file_names),num_of_examples_to_show, endpoint = True)))
    # print(save_epochs)

    image_lines = []
    for i in save_epochs:
        image_lines.append(get_examples_image_for_epoch(image_dir, i,image_size = image_size))
    # image_lines[0].show()
    # print('one image line: {}'.format(image_lines[0].size))
    total_height = (num_of_examples_to_show) * (image_size + padding) + padding
    total_width = image_lines[0].size[0]
    # print(total_height, total_width)

    new_im = Image.new('L',(total_width, total_height))

    y_offset = padding
    for im in image_lines:
      new_im.paste(im, (0,y_offset))
      y_offset += im.size[1] + padding

    # new_im.show()
    out_path = os.path.join(image_dir,'./../learning_developement.png')
    new_im.save(out_path)
    print('saving a epoch sequence image')

if __name__ == "__main__":

    # compare_test_acc_val()
    mydir = './{}'.format(sys.argv[1])
    mypath = os.path.abspath(mydir)


    make_time_image(mypath,128) 

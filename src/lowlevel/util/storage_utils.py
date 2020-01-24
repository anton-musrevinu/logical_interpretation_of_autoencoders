import pickle
import os
import csv
from torchvision.utils import save_image,make_grid
import torch
import numpy as np


def save_to_stats_pkl_file(experiment_log_filepath, filename, stats_dict):
    summary_filename = os.path.join(experiment_log_filepath, filename)
    with open("{}.pkl".format(summary_filename), "wb") as file_writer:
        pickle.dump(stats_dict, file_writer)


def load_from_stats_pkl_file(experiment_log_filepath, filename):
    summary_filename = os.path.join(experiment_log_filepath, filename)
    with open("{}.pkl".format(summary_filename), "rb") as file_reader:
        stats = pickle.load(file_reader)

    return stats

def clean_model_dir(model_save_dir, best_validation_model_idxs):
    for (dirpath, dirnames, filenames) in os.walk(model_save_dir):
            for file in filenames:
                model_idx_as_str = file.split('_')[0]
                if model_idx_as_str.isdigit() and not int(model_idx_as_str) in best_validation_model_idxs:
                    os.remove(os.path.join(dirpath,file))


def save_statistics(summary_filename, stats_dict, current_epoch, save_full_dict=False):
    """
    Saves the statistics in stats dict into a csv file. Using the keys as the header entries and the values as the
    columns of a particular header entry
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file
    :param stats_dict: the stats dict containing the data to be saved
    :param current_epoch: the number of epochs since commencement of the current training session (i.e. if the experiment continued from 100 and this is epoch 105, then pass relative distance of 5.)
    :param save_full_dict: whether to save the full dict as is overriding any previous entries (might be useful if we want to overwrite a file)
    :return: The filepath to the summary file
    """
    # summary_filename = os.path.join(experiment_log_dir, filename)
    current_epoch = current_epoch - 1

    mode = 'w' if ((current_epoch == 0) or (save_full_dict == True)) else 'a'
    with open(summary_filename, mode) as f:
        writer = csv.writer(f)
        if current_epoch == 0:
            writer.writerow(list(stats_dict.keys()))

        if save_full_dict:
            total_rows = len(list(stats_dict.values())[0])
            for idx in range(total_rows):
                row_to_add = [value[idx] for value in list(stats_dict.values())]
                writer.writerow(row_to_add)
        else:
            row_to_add = [value[current_epoch] for value in list(stats_dict.values())]
            writer.writerow(row_to_add)

    return summary_filename


def load_statistics(experiment_log_dir, filename):
    """
    Loads a statistics csv file into a dictionary
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file to load
    :return: A dictionary containing the stats in the csv file. Header entries are converted into keys and columns of a
     particular header are converted into values of a key in a list format.
    """
    summary_filename = os.path.join(experiment_log_dir, filename)

    with open(summary_filename, 'r+') as f:
        lines = f.readlines()

    keys = lines[0].split(",")
    stats = {key: [] for key in keys}
    for line in lines[1:]:
        values = line.split(",")
        for idx, value in enumerate(values):
            stats[keys[idx]].append(value)

    return stats

def _to_img(x):
    # print(x.min(), x.max())
    # x = 0.5 * (x + 1)
    # print(x.min(), x.max())
    # x = x.clamp(0, 1)
    x = x.reshape(-1,1,28,28)
    # if x.shape != (x.shape[0], 1, 28,28):
    #     raise Exception('output of the network is not in the right shape: {}'.format(x.shape))
    # if x.min() < 0 or x.max() > 1:
    #     raise Exception('values of the network are not between 0 and 1 min: {}, max: {}'.format(x.min(), x.max()))
    # x = x.view(x.size(0), 1, 28, 28)
    x = torch.Tensor(x).float()
    return x


def save_example_image(images_to_save, save_path_out, nrow = None):
        if type(images_to_save) == list and len(images_to_save) > 1:
            # print(images_to_save[0].shape)
            images = torch.from_numpy(np.concatenate(images_to_save, axis = -1))
            # print(images.shape)
        elif type(images_to_save) == list and len(images_to_save) == 1:
            images = images_to_save[0]
        else:
            images = images_to_save
        pic_target = images

        # pic_source = _to_img(source_image)
        # print(pic_target.shape, nrow)
        if nrow != None:
            save_image(pic_target, save_path_out, nrow = nrow, padding = 4)
        else:
            save_image(pic_target, save_path_out, padding = 4)



def save_feature_layer_example(model_save_dir, model_save_name, model_idx, feature_layer_ex, feature_layer_hidden_ex = None):
    save_file = os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx)))
    with open(save_file,'w') as f:
        f.write('feature_layer:\n')
        f.write('{}'.format(feature_layer_ex))
        if not issubclass(type(feature_layer_hidden_ex), type(None)):
            f.write('\nfeature_layer hidden representation:\n')
            f.write('{}'.format(feature_layer_hidden_ex))

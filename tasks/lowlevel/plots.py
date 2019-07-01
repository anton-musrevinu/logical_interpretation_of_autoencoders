import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from os import walk
import os

    # for k in total_losses.keys():
    #     if "train" in k:
    #         t = ax_1.plot(np.arange(len(total_losses[k])), total_losses[k], label=k)
    # ax_1.legend(loc=0)
    # ax_1.set_xlabel('Epoch number')
    
    
    # fig_2 = plt.figure(figsize=(8, 4))
    # ax_2 = fig_2.add_subplot(111)
    # for k in total_losses.keys():
    #     if "valid" in k:
    #         t = ax_2.plot(np.arange(len(total_losses[k])), total_losses[k], label=k)
    # ax_2.legend(loc=0)
    # ax_2.set_xlabel('Epoch number')


def plot_stats_in_graph(expirimentDir):

    total_losses = read_expiriment_data(expirimentDir)
    
    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in total_losses.keys():
        if "train" in k:
            ax_1.plot(np.arange(len(total_losses[k])), total_losses[k], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')
    
    
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in total_losses.keys():
        if "valid" in k:
            ax_2.plot(np.arange(len(total_losses[k])), total_losses[k], label=k)
            ax_2.legend(loc=0)
            ax_2.set_xlabel('Epoch number')

    plt.show()
    fig_1.savefig('{}/fig_loss.pdf'.format(expirimentDir))
    fig_2.savefig('{}/fig_acc.pdf'.format(expirimentDir))
    plt.close(fig_1)
    plt.close(fig_2)

def plot_stats_in_one_graph(expirimentDir, for_error):

    total_losses = read_expiriment_data(expirimentDir)

    colour_loss = 'firebrick'
    colour_acc = 'steelblue'

    linestyle_valid = '-'
    linestyle_trail = '--'
    
    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in total_losses.keys():
        if 'BCE' in k:
            if 'train' in k:
                ls = linestyle_trail
            else:
                ls = linestyle_valid
            ax_1.plot(np.arange(len(total_losses[k])), total_losses[k], label=k, c = colour_loss,ls = ls)
    ax_1.set_xlabel('Epoch number')
    ax_1.set_ylabel('BCE loss term', color = colour_loss)
    # ax_1.set_ylim([0,1.1])
    ax_2 = ax_1.twinx()
    
    for k in total_losses.keys():
        if "KLD" in k:
            if 'train' in k:
                ls = linestyle_trail
            else:
                ls = linestyle_valid
            ax_2.plot(np.arange(len(total_losses[k])), total_losses[k], label=k, c = colour_acc, ls = ls)
    ax_2.set_ylabel('KLD loss term', color = colour_acc)
    # ax_2.set_ylim([0.7,1])
    fig_1.legend(loc=4)
    # ax_2.set_xlabel('Epoch number')
    image_save_path = os.path.join(expirimentDir,'./trainging_loss_both.pdf'.format(for_error))
    print('saving figure at:', image_save_path)
    fig_1.savefig(image_save_path)
    plt.show()
    plt.close(fig_1)

def read_expiriment_data(expirimentDir):
    total_losses = {}
    indexes = {}
    with open('{}/result_outputs/summary.txt'.format(expirimentDir),'r') as f:
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
        print(key,total_losses[key])

    return total_losses

def compare_test_acc_val():

    f = []
    for (dirpath, dirnames, filenames) in walk('./'):
        f.extend(dirnames)
        break

    expirements = {}

    for dir_name in f:
        if not 'exp_' in str(dir_name):
            continue
        with open('./' + dir_name + '/result_outputs/test_summary.csv', 'r') as out:
            for i,line in enumerate(out):
                if i == 1:
                    expirements[dir_name] = [float(a) for a in line.split(',')]
        with open('./' + dir_name + '/info.txt') as out:
            for i, line in enumerate(out):
                if 'trainging time' in line:
                    training_time = float(line.split(':')[1].replace('\n',''))
                    expirements[dir_name].append(training_time)

    for exp, res in expirements.items():
        if len(res) == 2:
            print('expriment: {}\t, acc: {}\t, loss: {}'.format(exp, res[0], res[1]))
        elif len(res) == 3:
            print('expriment: {}\t, acc: {}\t, loss: {}\t, training time: {}s'.format(exp, res[0], res[1], res[2]))




if __name__ == "__main__":

    # compare_test_acc_val()
    # plot_stats_in_one_graph('./exp_14/') 
    plot_stats_in_one_graph(os.path.join(os.environ['HOME'],'./local_storage/backup_msc/output/experiments/ex_7_mnist_32_2/VAEManager/'), 'MSE')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from os import walk
import os
import sys

def plot_stats_in_graph(total_losses, expirimentDir):
    try:
        # total_losses = read_expiriment_data(expirimentDir)
        
        # Plot the change in the validation and training set error over training.
        fig_1 = plt.figure(figsize=(8, 4))
        ax_1 = fig_1.add_subplot(111)
        print(total_losses.keys())
        for k in total_losses.keys():
            if "train" in k:
                t = ax_1.plot(np.arange(len(total_losses[k])), total_losses[k], label=k)
        ax_1.legend(loc=0)
        ax_1.set_xlabel('Epoch number')
        
        
        fig_2 = plt.figure(figsize=(8, 4))
        ax_2 = fig_2.add_subplot(111)
        for k in total_losses.keys():
            if "valid" in k:
                t = ax_2.plot(np.arange(len(total_losses[k])), total_losses[k], label=k)
        ax_2.legend(loc=0)
        ax_2.set_xlabel('Epoch number')

        # plt.show()
        fig_1.savefig('{}/training_loss.pdf'.format(expirimentDir))
        fig_2.savefig('{}/training_acc.pdf'.format(expirimentDir))
        plt.close(fig_1)
        plt.close(fig_2)
    except Exception as e:
        print('Exception was cought: {}'.format(e))

def plot_stats_in_one_graph(expirimentDir):

    experiment_name = expirimentDir.split('/')[-1]

    total_losses = read_expiriment_data(expirimentDir)

    colour_train = 'firebrick'
    colour_val = 'steelblue'

    # colours = {'D_A': 'red', 'black', 'blue', 'brown', 'green', 'firebrick','steelblue'}

    linestyle_loss = '-'
    linestyle_acc = '--'

    ls = linestyle_loss
    
    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    # for k in total_losses.keys():
    #     if "train" in k:
    #         # if 'train' in k:
    #         #     ls = linestyle_trail
    #         # else:
    #         #     ls = linestyle_valid
    #         ax_1.plot(np.arange(len(total_losses[k])), total_losses[k], label=k, c = colour_train,ls = ls)
    ax_1.set_xlabel('Epoch number')
    # ax_1.set_ylabel('loss', color = colour_train)
    # ax_1.set_ylim([0,1.1])
    # ax_2 = ax_1.twinx()
    
    for idx,k in enumerate(total_losses.keys()):
        if "train" in k and not k == 'valid_G':
            if 'valid_G_acc' in k:
                colour = colour_val
            else:
                colour = colour_train
            if 'D' in k:
                ls = linestyle_acc
            else:
                ls = linestyle_loss
            k = k.replace('valid_','')
            # c = colours[k]
            ax_1.plot(np.arange(len(total_losses[k])), total_losses[k], label=k, ls = ls)
    ax_1.set_ylabel('acc', color = colour_val)
    # ax_1.set_yscale('log')
    # ax_2.set_ylim([0.7,1])
    fig_1.legend(loc=4)
    # ax_2.set_xlabel('Epoch number')
    plt.title(experiment_name)
    plt.show()
    fig_1.savefig('{}/fig_all_lin.pdf'.format(expirimentDir))
    plt.close(fig_1)

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
    mydir = './{}'.format(sys.argv[1])
    mypath = os.path.abspath(mydir)


    plot_stats_in_one_graph(mypath) 

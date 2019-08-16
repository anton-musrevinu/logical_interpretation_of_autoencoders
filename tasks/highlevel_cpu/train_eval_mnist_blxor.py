import os,sys
CURRENTDIR = os.path.dirname(os.path.realpath(__file__))
SRCDIR = os.path.abspath(os.path.join(CURRENTDIR, './../../src'))
sys.path.append(SRCDIR)


from experiment import *

experiment_parent_name = 'ex_7_mnist_32_2'
cluster_id = 'staff_compute'
task_type = 'blxor'
compress_fly = True

exp = Experiment(experiment_parent_name, cluster_id, task_type, compress_fly = compress_fly)

do_everything(exp)
# Same as:
# do_psdd_training(exp)
# encode_data(exp)
# do_classification_evaluation(exp)
# do_generative_query(exp, type_of_query = 'dis')
# do_analyse_feature_layer(exp, 1000)
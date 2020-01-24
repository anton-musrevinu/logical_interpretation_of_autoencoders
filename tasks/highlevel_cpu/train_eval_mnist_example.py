import os,sys
CURRENTDIR = os.path.dirname(os.path.realpath(__file__))
SRCDIR = os.path.abspath(os.path.join(CURRENTDIR, './../../src'))
sys.path.append(SRCDIR)


from experiment import *

experiment_parent_name = 'exp_mnist_test'
cluster_id = 'local'
task_type = 'classification'
compress_fly = False

exp = Experiment(experiment_parent_name, cluster_id, task_type, compress_fly = compress_fly)

do_everything(exp, testing = True)
# Same as:
# do_psdd_training(exp)
# encode_data(exp)
# do_classification_evaluation(exp)
# do_generative_query(exp, type_of_query = 'dis', testing = True)
# do_analyse_feature_layer(exp, 1000)
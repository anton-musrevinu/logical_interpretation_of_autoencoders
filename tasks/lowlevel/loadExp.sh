# python ./../src/pytorch_experiment_scripts/train_evaluate_emnist_autoencoder.py --experiment_name 'exp_1' --use_gpu False --feature_layer_size 512 --num_epochs 20 --bool_reg 0 --force_bool False --replace_existing False
NAME="./../results/exp_full_symetric_mnist_deep_2"
PROBFILE="./../../learnPSDD/EnsembleExperiments/exp_full_system_mnist_exp3/bernoulli_probs.txt"

python ./../src/load_feature_layer_emnist.py --experiment_name $NAME --model ConvAutoencoder_full_symetric_conv --prob_file $PROBFILE

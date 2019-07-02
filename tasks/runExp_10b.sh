#!/bin/bash
#
#SBATCH --job-name=ex_10_resnet_dropout
#SBATCH --output=ex_10_resnet_dropout.txt
#
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00


source activate mlp
python ../src/lowlevel/main.py --phase train --experiment_name ex_10_resnet_dropout --dataset mnist --gpu_ids 0,1 --feature_layer_size 32 --categorical_dim 2 --num_epochs 400 --ae_model_type resnet --no_dropout False
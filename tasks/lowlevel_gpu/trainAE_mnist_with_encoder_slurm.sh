#!/bin/bash
#
#SBATCH --job-name=job_exp_dropout_encoder
#SBATCH --output=job_exp_dropout_encoder.txt
#
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00


source activate mlp
python ../../src/lowlevel/main.py --phase train --experiment_name exp_dropout_encoder --dataset mnist --gpu_ids 0,1 --feature_layer_size 32 --categorical_dim 2 --num_epochs 400 --ae_model_type vanilla --use_dropout_decoder False --use_dropout_encoder True
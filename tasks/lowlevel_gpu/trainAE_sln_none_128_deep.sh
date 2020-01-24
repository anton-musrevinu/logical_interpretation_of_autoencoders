#!/bin/bash
#
#SBATCH --job-name=job_exp_sln_none_128_deep
#SBATCH --output=job_exp_sln_none_128_deep.txt
#
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=96:00:00


source activate mlp
python ../../src/lowlevel/main.py --phase train --experiment_name exp_sln_none_128_deep --dataset sln --gpu_ids 0,1 --feature_layer_size 128 --categorical_dim 2 --num_epochs 500 --batch_size 10 --ae_model_type deep --use_dropout_decoder False --use_dropout_encoder False
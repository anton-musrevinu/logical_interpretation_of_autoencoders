#!/bin/bash
#
#SBATCH --job-name=job_exp_sln_both_48
#SBATCH --output=job_exp_sln_both_48.txt
#
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00


source activate mlp
python ../../src/lowlevel/main.py --phase train --experiment_name exp_sln_both --dataset sln --gpu_ids 0,1 --feature_layer_size 48 --categorical_dim 2 --num_epochs 500 --batch_size 10 --ae_model_type vanilla
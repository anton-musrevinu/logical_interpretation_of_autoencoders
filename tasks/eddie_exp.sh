#!/bin/sh
# Grid Engine options (lines prefixed with #$)

-N psdd_search_fl16_c4_small              
-cwd ./msc/tasks          
-l h_rt=7200:00:00 
-l h_vmem=32G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh
 
# Load Python
module load anaconda
 
# Run the program
conda config --add envs_dirs ./miniconda3/envs/
source activate mlp

echo pwd

python experiment.py

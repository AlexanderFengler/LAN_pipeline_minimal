#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J data_generator

# priority
##SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output slurm/slurm_data_generator_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=40:00:00
#SBATCH --mem=16G
#SBATCH -c 12
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
##SBATCH --array=1-100

# --------------------------------------------------------------------------------------

# BASIC SETUP
source /users/afengler/.bashrc
conda deactivate
conda deactivate
conda activate lan_pipe

# Read in arguments:
config_dict_key=None
config_file=None

while [ ! $# -eq 0 ]
    do
        case "$1" in
            --config_file | -cf)
                config_file=$2
                ;;
            --config_dict_key | -cd)
                config_dict_key=$2
                ;;
        esac
        shift 2
    done

echo "The config file supplied is: $config_file"

python -u scripts/data_generation_script.py --config_file $config_file

#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J model_trainer

# priority
##SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output slurm/slurm_model_trainer_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -N 1

##SBATCH -p gpu --gres=gpu:1
##SBATCH --array=0-8 # should be 89

# --------------------------------------------------------------------------------------

# Setup
source /users/afengler/.bashrc
# module load cudnn/8.2.0
# module load cuda/11.7.1
# module load gcc/10.2
# module load graphviz/2.40.1

conda deactivate
conda deactivate
conda activate lan_pipe

# Read in arguments:
# These are supposed to be overwritten by arguments passed to the script
# they serve as reasonable defaults though
config_dict_key=None
config_path=None
networks_path="/users/afengler/data/proj_lan_pipeline/LAN_scripts/data/"
dl_workers=4
n_networks=2
backend="jax"
model="ddm"

echo "arguments passed to sbatch_network_training.sh $#"

while [ ! $# -eq 0 ]
    do
        case "$1" in
            --model | -m)
                echo "passing model as $2"
                model=$model
                ;;
            --config_path | -p)
                echo "passing config path $2"
                config_path=$2
                ;;
            --networks_path | -o)
                echo "passing output_folder $2"
                networks_path=$2
                ;;
            --n_networks | -n)
                echo "passing number of networks $2"
                n_networks=$2
                ;;
            --backend | -b)
                echo "passing deep learning backend specification: $2"
                backend=$2
                ;;
            --dl_workers | -d)
                echo "passing number of dataloader workers $2"
                dl_workers=$2
        esac
        shift 2
    done

echo "The config file supplied is: $config_path"
echo "The config dictionary key supplied is: $config_dict_key"
echo "Output folder is: $output_folder"

x='teststr' # defined only for the check below (testing whether SLURM_ARRAY_TASK_ID is set)
if [ -z ${SLURM_ARRAY_TASK_ID} ];
then
    for ((i = 1; i <= $n_networks; i++))
        do
            echo "NOW TRAINING NETWORK: $i of $n_networks"
            echo "No array ID"
            
            if [ "$backend" == "jax" ]; then
                python -u scripts/jax_training_script.py --model $model \
                                             --config_path $config_path \
                                             --config_dict_key 0 \
                                             --network_folder $networks_path \
                                             --dl_workers $dl_workers
            elif [ "$backend" == "torch" ]; then
                python -u scripts/torch_training_script.py --model $model \
                                                           --config_path $config_path \
                                                           --config_dict_key 0 \
                                                           --network_folder $networks_path \
                                                           --dl_workers $dl_workers
                                                       
            fi
        done
else
    for ((i = 1; i <= $n_networks; i++))
        do
            echo "NOW TRAINING NETWORK: $i of $n_networks"
            echo "No array ID"
            
            if [ "$backend" == "jax" ]; then
                python -u scripts/jax_training_script.py --model $model \
                                                         --config_path $config_path \
                                                         --config_dict_key $SLURM_ARRAY_TASK_ID \
                                                         --network_folder $networks_path \
                                                         --dl_workers $dl_workers
            elif [ "$backend" == "torch" ]; then
                python -u scripts/torch_training_script.py --model $model \
                                                           --config_path $config_path \
                                                           --config_dict_key $SLURM_ARRAY_TASK_ID \
                                                           --network_folder $networks_path \
                                                           --dl_workers $dl_workers
                                                       
            fi
        done
fi
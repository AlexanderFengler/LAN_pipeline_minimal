#!/bin/bash

# Run configs (parameters to script) ----------------
model="ddm" # "ddm_deadline"
#'ds_conflict_drift'

# Network training configs -----------
network_type="cpn" # cpn or lan
optimizer="adam" # which optimizer to choose
backend="jax" # jax, torch
partition="gpu" # gpu, cpu
dl_workers=2 # number of processes to use for data-loading

#network_n_epochs=20 # How many epochs to train the network for ?
data_gen_n_samples_per_sim=20000 #200000 # How many simulated trials per call to the simulator ?
data_gen_n_parameter_sets=1000 #5000 # How many parameter sets do we request ? 
data_gen_n_training_examples_per_parameter_set=2000 # How many training examples do we harvest from a given parameter set

project_folder='/users/afengler/data/proj_lan_pipeline/LAN_scripts/'

network_training_config_path=$project_folder'/data/config_files/network/'\
$network_type'/'$model'/train_config_opt_'$optimizer'_n_'$data_gen_n_samples_per_sim'_dt_0.001_nps_'\
$data_gen_n_parameter_sets'_npts_'$data_gen_n_training_examples_per_parameter_set'_architecture_search.pickle'

networks_path=$project_folder'/data/networks/'$network_type'/'$backend'/'$model'/'

echo 'Config file passed to sbatch_network_training.sh'
echo $network_training_config_path
echo $networks_path

# Train networks ----
if [ "$partition" == "gpu" ]; then
    sbatch -p gpu --gres=gpu:1 \
                  --account=carney-frankmj-condo \
                  --array=0-5 sbatch_scripts/sbatch_network_training.sh \
                  --model $model \
                  --backend $backend \
                  --config_path $network_training_config_path \
                  --networks_path $networks_path \
                  --dl_workers $dl_workers
                    
elif [ "$partition" == "cpu" ]; then
    sbatch -p batch --account=carney-frankmj-condo  \
                    --array=0-2 sbatch_scripts/sbatch_network_training.sh \
                    --model $model \
                    --backend $backend \
                    --config_path $network_training_config_path \
                    --networks_path $networks_path \
                    --dl_workers $dl_workers
                    
    sbatch -p batch --array=2-4 sbatch_scripts/sbatch_network_training.sh \
                    --model $model \
                    --backend $backend \
                    --config_path $network_training_config_path \
                    --networks_path $networks_path \
                    --dl_workers $dl_workers                  
fi
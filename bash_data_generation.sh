#!/bin/bash

# Run configs (parameters to script) ----------------
oscar_acct="carney-frankmj-condo" # your oscar account name (CHECK BELOW WHETHER YOU CAN USE TWO ACCOUNTS)
also_use_base_acct=true # whether to use your base account as well (if you have one / if it has any relevant computational resources)
n_base_acct=150 # how many jobs to run with your base account 
n_oscar_acct=150 # how many jobs to run with your oscar account

model='ddm' # valid model string supported by ssms package # 'ds_conflict_drift_angle'

project_folder='/users/afengler/data/proj_lan_pipeline/LAN_scripts/' # base folder of your project

while [ ! $# -eq 0 ]
    do
        case "$1" in
             --model | -m)
                model=$2
             ;;
             --project_folder | -p)
                project_folder=$2
             ;;
        esac
        shift 2
    done

# Data generation configs --------
# NOTE: The data generation configs are also used to identify (create) the data folder

# How many simulated trials per call to the simulator?
data_gen_n_samples_per_sim=20000 #200000 #200000 

# How many parameter sets do we request? 
data_gen_n_parameter_sets=1000 #5000   #5000

# How many training examples do we harvest from a given parameter set?
data_gen_n_training_examples_per_parameter_set=2000 # this is not relevant for cpu_only

# Data generator approach (lan, cpn_only)
data_generator_approach='lan'

data_generation_config_file=$project_folder'/data/config_files/data_generation/'$data_generator_approach'/'$model'/'\
'nsim_'$data_gen_n_samples_per_sim'_dt_0.001_nps_'$data_gen_n_parameter_sets\
'_npts_'$data_gen_n_training_examples_per_parameter_set'.pickle'

# NOTE: The run is split into two accts (because I, Alex, have two)
# This setting may or may not help you. 

# Run with personal account
if [ $also_use_base_acct ]; then
    sbatch -p batch --array=0-$n_base_acct sbatch_scripts/sbatch_data_generation.sh --config_file $data_generation_config_file | cut -f 4 -d' '
fi 

# Run with frankmj account
sbatch -p batch --account=$oscar_acct --array=0-$n_oscar_acct sbatch_scripts/sbatch_data_generation.sh --config_file $data_generation_config_file | cut -f 4 -d' '
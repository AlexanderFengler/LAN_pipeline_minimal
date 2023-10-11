#!/bin/bash
# Run configs (parameters to script) ----------------
model='ddm'
#'ds_conflict_drift_angle'
project_folder='/users/afengler/data/proj_lan_pipeline/LAN_scripts/'
#'/users/afengler/data/proj_dynddm/dynddm'

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

# Run with personal account
sbatch -p batch --array=0-150 sbatch_scripts/sbatch_data_generation.sh --config_file $data_generation_config_file | cut -f 4 -d' '

# Run with frankmj account
sbatch -p batch --account=carney-frankmj-condo --array=0-150 sbatch_scripts/sbatch_data_generation.sh --config_file $data_generation_config_file | cut -f 4 -d' '
# Append system path to include the config scripts
import sys
import os
from copy import deepcopy
import lanfactory
import ssms
import hddm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config import *
import tensorflow
import torch
import pandas as pd
import argparse

def none_or_str(value):
    print('none_or_str')
    print(value)
    print(type(value))
    if value == 'None':
        return None
    return value

def none_or_int(value):
    print('none_or_int')
    print(value)
    print(type(value))
    #print('printing type of value supplied to non_or_int ', type(int(value)))
    if value == 'None':
        return None
    return int(value)

if __name__ == "__main__":
    
    # Interface ----
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--networks_path",
                     type = none_or_str,
                     default = None)
    CLI.add_argument("--model",
                     type = none_or_str,
                     default = None)
    CLI.add_argument("--param_recov_n_data_sets",
                     type = int,
                     default = 1000)
    CLI.add_argument("--param_recov_n_subjects",
                     type = int,
                     default = 10)
    CLI.add_argument("--param_recov_n_trials_per_subject",
                     type = int,
                     default = 1000)
    CLI.add_argument("--param_recov_n_lans_to_pick",
                     type = int,
                     default = 10)
    CLI.add_argument("--param_recov_n_burn",
                     type = int,
                     default = 1000)
    CLI.add_argument("--param_recov_n_mcmc",
                     type = int,
                     default = 5000)
    CLI.add_argument("--param_recov_n_chains",
                     type = int,
                     default = 2)
    
    args = CLI.parse_args()
    print('Arguments passed: ', args)
        
    my_config_generator = central_configurator(model = args.model,
                                               param_recov_n_data_sets = args.param_recov_n_data_sets,
                                               param_recov_n_subjects = args.param_recov_n_subjects,
                                               param_recov_n_trials_per_subject = args.param_recov_n_trials_per_subject,
                                               param_recov_n_lans_to_pick = args.param_recov_n_lans_to_pick,
                                               param_recov_n_burn = args.param_recov_n_burn,
                                               param_recov_n_mcmc = args.param_recov_n_mcmc,
                                               param_recov_n_chains = args.param_recov_n_chains)
    
    my_config_generator._make_param_recov_configs(networks_path = args.networks_path,
                                                  show_top_n = 10)
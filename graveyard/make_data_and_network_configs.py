# Append system path to include the config scripts
import sys
import os
from copy import deepcopy

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config import *
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
    CLI.add_argument("--data_gen_n_samples_per_sim",
                     type = int,
                     default = 200000)
    CLI.add_argument("--data_gen_n_parameter_sets",
                     type = int,
                     default = 5000)
    CLI.add_argument("--data_gen_n_training_examples_per_parameter_set",
                     type = int,
                     default = 2000)
    CLI.add_argument("--network_n_epochs",
                     type = int,
                     default = 30)
    
    
    args = CLI.parse_args()
    print('Arguments passed: ', args)
        
    my_config_generator = central_configurator(model = args.model,
                                               data_gen_n_samples_per_sim = args.data_gen_n_samples_per_sim,
                                               data_gen_n_parameter_sets = args.data_gen_n_parameter_sets,
                                               data_gen_n_training_examples_per_parameter_set = args.data_gen_n_training_examples_per_parameter_set,
                                               network_n_epochs = args.network_n_epochs
                                              )
    
    my_config_generator.make_config_files(make_config_param_recov = 0,
                                          make_config_data_gen = 1,
                                          make_config_network = 1
                                         )
# Load modules
import hddm
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy as scp
import psutil
from time import time
from copy import deepcopy
import os
import pickle
import argparse
from multiprocessing import Pool
from functools import partial
from hddm.simulators.hddm_dataset_generators import simulator_h_c
import torch

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        except:
            print('SOMETHING WENT WRONG WITH SAMPLING ....')
        finally:
            sys.stdout = old_stdout

def none_or_str(value):
    print('none_or_str')
    print(value)
    print(type(value))
    if value == 'None':
        return None
    return value

def prepend_str_with_zeros(str_in = '',
                           len_out = 5):
    
    if type(str_in) is not str:
        str_in = str(str_in)
    
    len_in = len(str_in)
    len_diff = len_out - len_in
    assert len_diff >= 0, 'Supplied a string that is longer than len_out'
    
    if len_diff == 0:
        return str_in
    
    else:
        for i in range(len_diff):
            str_in = '0' + str_in
    return str_in

def run_model(chain_idx,
              recov_idx = 0,
              run_config = {},
              parallel = 0,
              model_idx = 0,
              verbose = 0,
              hddm_basic = 0):
       
    # ASSIGN MODEL CONFIG
    model_config = run_config['model_config']
    
    print('MODEL CONFIG for MODEL IDX: ', model_idx)
    print(model_config)
    
    # LOAD PARAMETER RECOVERY DATA
    data = pickle.load(open(run_config['parameter_recovery_data_loc'], 'rb'))
    parameter_dict = data[recov_idx]['parameter_dict']
    data = data[recov_idx]['data']

    # LOAD NETWORK
    if not hddm_basic:
        network_config = pickle.load(open(run_config['lan_config_files'][model_idx], 'rb'))
        network_path = run_config['lan_files'][model_idx]
        network = hddm.torch.mlp_inference_class.LoadTorchMLPInfer(model_file_path = network_path,
                                                                   network_config = network_config,
                                                                   input_dim = len(model_config["params"]) + 2)
    
    # ACTUAL RUN  ---------------------------------------------------
    start_t = time()
    if not hddm_basic:
        model_ = hddm.HDDMnn(data,
                             model_config = model_config,
                             network = network,
                             include = model_config["params"],
                             is_group_model = True)
        
    else:
        model_ = hddm.HDDMnn(data,
                             model = run_config['model_name'],
                             include = model_config["params"],
                             is_group_model = True)

    recov_idx_str = prepend_str_with_zeros(str_in = str(recov_idx),
                                           len_out = 5)
    chain_idx_str = prepend_str_with_zeros(str_in = str(chain_idx),
                                           len_out = 2)
    
    sample_on_gpu = int(torch.cuda.device_count() > 0)
    sample_on_gpu_str = str(sample_on_gpu)
    
    custom_ = int(not hddm_basic)
    custom_str = str(custom_)
    
    db_file_name = run_config['save_folder'] + '/' + \
                        run_config['lan_ids'][model_idx] + '_db_' + 'custom_' + custom_str +  '_gpu_' + sample_on_gpu_str + \
                            '_recovid_' + recov_idx_str + '_chain_' + chain_idx_str + '.db'

    model_file_name = run_config['save_folder'] + '/' + \
                        run_config['lan_ids'][model_idx] + '_model_' + 'custom_' + custom_str +  '_gpu_' + sample_on_gpu_str + \
                            '_recovid_' + recov_idx_str + '_chain_' + chain_idx_str + '.pickle'

    # Check if model file name exists --> if yes don't run anything
    if os.path.exists(model_file_name):
        print(model_file_name, ' EXISTS --> NOT RUNNING THIS MODEL')
        return None, None
    else:
        try:
            print('STARTING SAMPLING')
            if not verbose:
                with suppress_stdout():
                    model_.sample(run_config['n_mcmc'] + run_config['n_burn'], run_config['n_burn'], 
                                      dbname = db_file_name, 
                                      db = 'pickle')
            else:
                model_.sample(run_config['n_mcmc'] + run_config['n_burn'], run_config['n_burn'],
                                  dbname = db_file_name,
                                  db = 'pickle')

            print('FINISHED SAMPLING')
            time_elapsed = time() - start_t
            print('TIME ELAPSED')
            print(str(time_elapsed), ' seconds')
            print('NOW SAVING MODEL')
            model_.save(model_file_name)
        except:
            print("Something went wrong with sampling from the model")
            print("Model_id: ", run_config['lan_ids'][model_idx])
            return -1

        # Make dataframe containing info about model
        model_data = pd.DataFrame(columns = ['model_idx', # number of the model
                                             'model_id', # string indentifier of model
                                             'network_hddm_included',
                                             'recov_idx', 'recov_idx_str',
                                             'chain_idx', 'chain_idx_str',
                                             'model_file', 'db_file', 
                                             'data', 'reg_models',
                                             'parameter_dict', 'dic', 
                                             'traces', 'time', 'gpu'])
        
        if not hddm_basic:
            model_id_tmp = run_config['lan_config_files'][model_idx]
            network_hddm_included_tmp = 0
        else:
            model_id_tmp = '00000000000000000000000000000000'
            network_hddm_included_tmp = 1
            
        model_data = model_data.append({'model_idx': model_idx, 
                                            'model_id': model_id_tmp,
                                            'network_hddm_included': network_hddm_included_tmp,
                                            'recov_idx': recov_idx, 'recov_idx_str': recov_idx_str,
                                            'chain_idx': chain_idx, 'chain_idx_str': chain_idx_str,
                                            'model_file': model_file_name, 'db_file': db_file_name,
                                            'data': data, 'reg_models': {},
                                            'parameter_dict': parameter_dict,
                                            'dic': model_.dic,
                                            'traces': pd.DataFrame.from_dict({key: model_.mc.db._traces[key].gettrace() for key in model_.mc.db._traces.keys()}),
                                            'time': time_elapsed,
                                            'gpu': sample_on_gpu},
                                            ignore_index = True)

        data_file_name = run_config['save_folder'] + '/' + \
                            run_config['lan_ids'][model_idx] + '_df_' + 'custom_' + custom_str +  '_gpu_' + sample_on_gpu_str + \
                                '_recovid_' + recov_idx_str + '_chain_' + chain_idx_str + '.pickle'              

        pickle.dump(model_data, open(data_file_name, 'wb'))

    if parallel:
        return 1
    else:
        return model_, model_data
    
    # WRITE ONE MORE OF THESE THAT ALLOWS ARBITRARY REGRESSION MODEL
    # THINK ABOUT HOW TO STORE THAT !

if __name__ == "__main__":
    
    # Interface ----
    CLI = argparse.ArgumentParser()

    CLI.add_argument('--config_file',
                     type = none_or_str,
                     default = None)
    CLI.add_argument('--model_idx',
                     type = int,
                     default = 0)
    CLI.add_argument('--recov_idx',
                     type = int,
                     default = 0)
    CLI.add_argument('--parallel',
                     type = int,
                     default = 0)
    CLI.add_argument('--verbose',
                     type = int,
                     default = 0)
    CLI.add_argument('--hddm_basic',
                     type = int,
                     default = 0)
    
    args = CLI.parse_args()
    print('Arguments passed: ', args)
    
    run_config = pickle.load(open(args.config_file, 'rb'))
    
    print('LOADED FROM MODEL SPEC: ', args.model_idx)

    # RUN MODELS ------------------------------------------------------------------
    if not args.parallel:
        
        print('STARTING RECOVERY: ', str(args.recov_idx))
        for chain_idx in range(run_config['n_chains']):
            model_, model_data = run_model(chain_idx = chain_idx,
                                           recov_idx = args.recov_idx,
                                           run_config = run_config,
                                           parallel = 0,
                                           model_idx = args.model_idx,
                                           verbose = args.verbose,
                                           hddm_basic = args.hddm_basic,
                                           )
        print('RECOVERY FINISHED FOR: ', str(recov_idx))
    else:
        
        n_cpus = psutil.cpu_count(logical = False)
        if n_cpus < run_config['n_chains']:
            n_processes = n_cpus
        else:
            n_processes = run_config['n_chains']
            
        print('N PROCESSES ASSIGNED: ', n_processes)
        
        chain_idxs = [i for i in range(run_config['n_chains'])]
        
        prepped_run_model = partial(run_model,
                                     run_config = run_config,
                                     parallel = 1,
                                     model_idx = args.model_idx,
                                     recov_idx = args.recov_idx,
                                     verbose = args.verbose,
                                     hddm_basic = args.hddm_basic
                                   )

        # Starting pool
        with Pool(processes = n_processes) as pool:
            pool_out = pool.map(prepped_run_model, chain_idxs)
        
        print('POOL OUT: ', pool_out)
        if (-1) not in pool_out:
            print('SCRIPT FINISHED: SUCCESSFULLY')
        else:
            print('SCRIPT FINISHED: SOME MODELS EITHER CREATED PROBLEMS OR DIDNT RUN')

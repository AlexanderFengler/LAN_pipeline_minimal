# Append system path to include the config scripts
import sys
import os
from copy import deepcopy

print('importing lanfactory')
import lanfactory

print('importing ssms')
import ssms

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config import *
from config import model_performance_utils

import torch

import pandas as pd

import pickle

class central_configurator():
    def __init__(self,
                 model = 'ddm',
                 data_gen_n_samples_per_sim = 200000,
                 data_gen_n_parameter_sets = 5000,
                 data_gen_n_training_examples_per_parameter_set = 2000,
                 data_gen_delta_t = 0.001,
                 network_n_epochs = 20,
                 network_cpu_batch_size = 10000,
                 network_gpu_batch_size = 100000,
                 param_recov_n_subjects = 10,
                 param_recov_n_trials_per_subject = 1000,
                 param_recov_n_lans_to_pick = 10,
                 param_recov_n_burn = 2000,
                 param_recov_n_mcmc = 6000,
                 param_recov_n_chains = 2,
                 **kwargs,
                ):
        
        # BASIC PARAMETERS --------------
        # Specify model
        print(locals())
        local_variables = locals()
        self.model = model
        #self.file_identifier = file_identifier # kwargs.pop('file_identifier', 'ddm_training_data')
        self.base_folder = kwargs.pop('base_folder', '/users/afengler/data/proj_lan_pipeline/LAN_scripts/')
        # -------------------------------
        
        # Where do you want to save the config file?
        self.config_save_folder = kwargs.pop('config_save_folder', '/users/afengler/data/' + \
                                                                     'proj_lan_pipeline/LAN_scripts/config_files/')

        # DATA GENERATOR PART
        self.data_gen_approach = kwargs.pop('data_gen_approach', 'lan') # Training data for what kind of likelihood approimator?
        self.data_gen_network_type = kwargs.pop('data_gen_network_type', 'mlp') # Type of network to train

        # Specify arguments which you want to adjust in the data generator
        self.data_gen_arg_dict =  {'dgp_list': model,
                                   'n_samples': data_gen_n_samples_per_sim,
                                   'n_parameter_sets': data_gen_n_parameter_sets,
                                   'delta_t': data_gen_delta_t,
                                   'n_training_samples_by_parameter_set': data_gen_n_training_examples_per_parameter_set,
                                   'n_subruns': 5}

        self.data_gen_model_config_arg_dict = kwargs.pop('data_gen_model_config', {})
        assert type(self.data_gen_model_config_arg_dict), 'supplied model config for data generator is not a dictionary'

        # Name of the config file
        self.data_gen_config_save_name = self.model + '_nsim_' + str(self.data_gen_arg_dict['n_samples']) + \
                                            '_dt_' + str(self.data_gen_arg_dict['delta_t']) + \
                                                '_nps_' + str(self.data_gen_arg_dict['n_parameter_sets']) + \
                                                    '_npts_' + str(self.data_gen_arg_dict['n_training_samples_by_parameter_set']) + \
                                                        '.pickle'

        # NETWORK PART
        self.network_train_config_save_folder = self.config_save_folder

        # Specify training data folder:
        self.network_training_data_folder = self.base_folder + 'data/' + \
                                        self.data_gen_approach + '_' + self.data_gen_network_type + \
                                            '/training_data_0_nbins_0_n_'  + str(self.data_gen_arg_dict['n_samples']) + \
                                                '/' + self.model + '/'

        # Specify the name of the config file
        self.network_dl_backend = kwargs.pop('network_dl_backend', 'torch')
        
        self.network_train_config_save_name = self.network_dl_backend + '_network_train_config_' + self.model + \
                                                '_nsim_' + str(self.data_gen_arg_dict['n_samples']) + \
                                                    '_dt_' + str(self.data_gen_arg_dict['delta_t']) + \
                                                        '_nps_' + str(self.data_gen_arg_dict['n_parameter_sets']) + '_npts_' + \
                                                            str(self.data_gen_arg_dict['n_training_samples_by_parameter_set']) + \
                                                                '_architecture_search.pickle'

        # How many epochs to train?
        self.network_n_epochs = network_n_epochs

        # Network architectures
        self.network_layer_sizes = [[100, 100, 100, 1], 
                            [100, 100, 100, 100, 1], 
                            [100, 100, 100, 100, 100, 1],
                            [120, 120, 120, 1], 
                            [120, 120, 120, 120, 1], 
                            [120, 120, 120, 120, 120, 1],
                            [150, 150, 150, 1], 
                            [150, 150, 150, 150, 1], 
                            [150, 150, 150, 150, 150, 1]
                           ]

        self.network_layer_types = [['dense', 'dense', 'dense', 'dense'], 
                            ['dense', 'dense', 'dense', 'dense', 'dense'], 
                            ['dense', 'dense', 'dense', 'dense', 'dense', 'dense'],
                            ['dense', 'dense', 'dense', 'dense'], 
                            ['dense', 'dense', 'dense', 'dense', 'dense'], 
                            ['dense', 'dense', 'dense', 'dense', 'dense', 'dense'],
                            ['dense', 'dense', 'dense', 'dense'], 
                            ['dense', 'dense', 'dense', 'dense', 'dense'], 
                            ['dense', 'dense', 'dense', 'dense', 'dense', 'dense'],
                           ]

        self.network_activations = [['tanh', 'tanh', 'tanh', 'linear'], 
                            ['tanh', 'tanh', 'tanh', 'tanh', 'linear'], 
                            ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'linear'],
                            ['tanh', 'tanh', 'tanh', 'linear'], 
                            ['tanh', 'tanh', 'tanh', 'tanh', 'linear'], 
                            ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'linear'],
                            ['tanh', 'tanh', 'tanh', 'linear'], 
                            ['tanh', 'tanh', 'tanh', 'tanh', 'linear'], 
                            ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'linear'],
                           ]

        # Train / validations split
        self.network_train_val_split = [0.98, 0.98, 0.98, 
                                        0.98, 0.98, 0.98,
                                        0.98, 0.98, 0.98]

        # Training files
        self.network_n_training_files = [1000]

        self.network_cpu_batch_size = network_cpu_batch_size
        self.network_gpu_batch_size = network_gpu_batch_size

        # PARAMETER RECOVERY
        # Specify configs for parameter recovery dataset
        self.param_recov_n_data_sets = 1000
        self.param_recov_n_subjects = param_recov_n_subjects
        self.param_recov_n_trials_per_subject = param_recov_n_trials_per_subject #kwargs.pop('param_recov_n_trials_per_subject', 1000)

        self.param_recov_save_folder  = self.base_folder + 'data/parameter_recovery/' + \
                                            self.model + '/' + 'subj_' + str(self.param_recov_n_subjects) + \
                                                '_trials_' + str(self.param_recov_n_trials_per_subject)

        self.param_recov_save_file_name = '/' + self.model + '_parameter_recovery_base_dataset_subj_' + \
                                            str(self.param_recov_n_subjects) + '_ntrials_' + \
                                                str(self.param_recov_n_trials_per_subject) + '.pickle'

        self.param_recov_data_file = self.param_recov_save_folder + self.param_recov_save_file_name

        # Get top n models here
        self.param_recov_n_lans_to_pick = 10

        # PARAMETER RECOVERY HYPERPARAMETERS

        # MCMC specifics
        self.param_recov_n_burn = param_recov_n_burn
        self.param_recov_n_mcmc = param_recov_n_mcmc
        self.param_recov_n_chains = param_recov_n_chains

        # Other metadata
        if model in ssms.config.model_config.keys(): # , 'Invalid model choice for parameter recovery study'
            self.param_recov_model_config =  deepcopy(ssms.config.model_config[model])
        else:
            print('Model is not part of ssm-simulators package --> no parameter recovery study possible')

    def _make_data_generator_configs(self):
        print('Making generator config')
        print('save name: ', self.data_gen_config_save_name)
        print('save folder: ', self.config_save_folder)
        make_data_generator_configs(model = self.model,
                                    generator_approach = self.data_gen_approach,
                                    data_generator_arg_dict = self.data_gen_arg_dict,
                                    model_config_arg_dict = self.data_gen_model_config_arg_dict,
                                    save_name = self.data_gen_config_save_name,
                                    save_folder = self.config_save_folder)
        return 1

    def _make_network_configs(self):
        # Loop objects
        config_dict = {}
        network_arg_dicts = {}
        train_arg_dicts = {}    
        cnt = 0
        print('Making network config')
        print('save name: ', self.network_train_config_save_name)
        print('save folder: ', self.network_train_config_save_folder)

        for i in range(len(self.network_layer_sizes)):
            for j in range(len(self.network_n_training_files)):
                network_arg_dict = {'layer_types': self.network_layer_types[i],
                                    'layer_sizes': self.network_layer_sizes[i],
                                    'activations': self.network_activations[i],
                                    'loss': ['huber'],
                                    'model_id': self.model
                                    }

                train_arg_dict = {'n_epochs': self.network_n_epochs,
                                  'n_training_files': self.network_n_training_files[j],
                                  'train_val_split': self.network_train_val_split[i],
                                  'cpu_batch_size': self.network_cpu_batch_size,
                                  'gpu_batch_size': self.network_gpu_batch_size,
                                  'shuffle_files': True,
                                  'label_prelog_cutoff_low': 1e-7,
                                  'label_prelog_cutoff_high': None,
                                  'save_history': True,
                                  'callbacks': ['checkpoint', 'earlystopping', 'reducelr'],
                                  }

                config_dict[cnt] = make_train_network_configs(training_data_folder=self.network_training_data_folder,
                                                              training_file_identifier=self.model,
                                                              save_folder = self.network_train_config_save_folder,
                                                              train_val_split=self.network_train_val_split[i],
                                                              network_arg_dict = network_arg_dict,
                                                              train_arg_dict = train_arg_dict,
                                                              save_name = self.network_train_config_save_name)

                print('NEW PRINT')
                print(cnt)
                cnt += 1

        print('Now saving')
        pickle.dump(config_dict, open(self.network_train_config_save_folder + self.network_train_config_save_name, 'wb'))
        print(self.network_train_config_save_folder + self.network_train_config_save_name)
        
        return 1
    
    def _make_param_recov_configs(self,
                                  networks_path = '/users/afengler/data/proj_lan_pipeline/LAN_scripts/data/torch_models',
                                  show_top_n = 10):
        
        # Check if parameter_recovery_dataset exists
        # If it doesn't we generate parameter recovery data
        
        self.param_recov_network_data = model_performance_utils.get_model_performance_summary_df(filter_ = self.model,
                                                                                                 path = networks_path)
        self.param_recov_lan_files = self.param_recov_network_data.loc[:, 'model_path'].to_list()[:self.param_recov_n_lans_to_pick]
        self.param_recov_lan_config_files = self.param_recov_network_data.loc[:, 'network_config_path'].to_list()[:self.param_recov_n_lans_to_pick]
        self.param_recov_lan_ids = self.param_recov_network_data.loc[:, 'model_id'].to_list()[:self.param_recov_n_lans_to_pick]

        if not os.path.exists(self.param_recov_data_file):
            __ = make_parameter_recovery_dataset(model = self.model,
                                                 save_folder = self.param_recov_save_folder,
                                                 save_file = self.param_recov_save_file_name,
                                                 n_subjects = self.param_recov_n_subjects,
                                                 n_trials_per_subject = self.param_recov_n_trials_per_subject)

        make_param_recovery_configs(model_name = self.model,
                                    parameter_recovery_data_loc = self.param_recov_data_file,
                                    lan_files = self.param_recov_lan_files,
                                    lan_ids = self.param_recov_lan_ids,
                                    lan_config_files = self.param_recov_lan_config_files,
                                    save_folder = self.param_recov_save_folder,
                                    model_config = self.param_recov_model_config,
                                    n_burn = self.param_recov_n_burn,
                                    n_mcmc = self.param_recov_n_mcmc,
                                    n_chains = self.param_recov_n_chains)
        return 1
    
    
    def make_config_files(self,           
                          make_config_param_recov=0,
                          make_config_data_gen=1,
                          make_config_network=1,
                          param_recov_networks_path = '/users/afengler/data/proj_lan_pipeline/LAN_scripts/data/torch_models',
                          param_recov_show_top_n = 10):
        
        if make_config_data_gen:
            print('Making config for data generator')
            self._make_data_generator_configs()
        if make_config_network:
            print('Making config for networks')
            self._make_network_configs()
        if make_config_param_recov:
            print('Making config for parameter recovery')
            self._make_config_param_recov(networks_path = param_recov_networks_path,
                                          show_top_n = param_recov_show_top_n)
        return
    
def make_data_generator_configs(model = 'ddm',
                                generator_approach = 'lan',
                                data_generator_arg_dict = None,
                                model_config_arg_dict = None,
                                save_name = None,
                                save_folder = ''):
    
    # Load copy of the respective model's config dict from ssms
    model_config = deepcopy(ssms.config.model_config[model])
    
    # Load copy of the respective data_generator_config dicts 
    data_config = deepcopy(ssms.config.data_generator_config[generator_approach])
    data_config['dgp_list'] = model
    
    for key, val in data_generator_arg_dict.items():
        data_config[key] = val
        
    for key, val in model_config_arg_dict.items():
        model_config[key] = val

    config_dict = {'model_config': model_config, 'data_config': data_config}
    
    if save_name is not None:
        if len(save_folder) > 0:
            
            if save_folder[-1] == '/':
                pass
            else:
                save_folder = save_folder + '/'
        
        # Create save_folder if not already there
        lanfactory.utils.try_gen_folder(folder = save_folder, 
                                        allow_abs_path_folder_generation = True)
                
        # Dump pickle file
        pickle.dump(config_dict, open(save_folder + save_name, 'wb'))
        
        print('Saved to: ')
        print(save_folder + save_name)
    
    return {'config_dict':config_dict, 
            'config_file_name': None if save_name is None else save_folder + save_name}
    
def make_train_network_configs(training_data_folder = None,
                               train_val_split = 0.9, 
                               save_folder = '',
                               network_arg_dict = None,
                               train_arg_dict = None,
                               save_name = None):
    
    # Load 
    train_config = deepcopy(lanfactory.config.train_config_mlp)
    network_config = deepcopy(lanfactory.config.network_config_mlp)
    
    for key, val in network_arg_dict.items():
        network_config[key] = val
        
    for key, val in train_arg_dict.items():
        train_config[key] = val
    
    config_dict = {'network_config': network_config,
                   'train_config': train_config,
                   'training_data_folder': training_data_folder,
                   'train_val_split': train_val_split}
    
    if save_name is not None:
        if len(save_folder) > 0:
            
            if save_folder[-1] == '/':
                pass
            else:
                save_folder = save_folder + '/'
        
        # Create save_folder if not already there
        lanfactory.utils.try_gen_folder(folder = save_folder, 
                                        allow_abs_path_folder_generation = True)
             
        # Dump pickle file
        print('Saved to: ')
        print(save_folder + save_name)
        
        pickle.dump(config_dict, open(save_folder + save_name, 'wb'))
    
    return {'config_dict': config_dict, 
            'config_file_name': None if save_name is None else save_folder + save_name}
        
def make_param_recovery_configs(model_name = 'ddm',
                                parameter_recovery_data_loc = '',
                                lan_files = [],
                                lan_config_files = [],
                                lan_ids = [],
                                save_folder = '',
                                model_config = None,
                                n_burn = 1000,
                                n_mcmc = 5000,
                                n_chains = 4):
    
    parameter_recovery_config_dict = {}
    parameter_recovery_config_dict['parameter_recovery_data_loc'] = parameter_recovery_data_loc
    parameter_recovery_config_dict['model_config'] = model_config
    parameter_recovery_config_dict['model_name'] = model_name
    parameter_recovery_config_dict['lan_files'] = lan_files
    parameter_recovery_config_dict['lan_ids'] = lan_ids
    parameter_recovery_config_dict['lan_config_files'] = lan_config_files
    parameter_recovery_config_dict['n_burn'] = n_burn
    parameter_recovery_config_dict['n_mcmc'] = n_mcmc
    parameter_recovery_config_dict['n_chains'] = n_chains
    parameter_recovery_config_dict['save_folder'] = save_folder
    parameter_recovery_config_dict['save_file'] = save_folder + '/' + model_name + \
                     '_parameter_recovery_run_config.pickle'
    
    lanfactory.utils.try_gen_folder(folder = save_folder, 
                                    allow_abs_path_folder_generation = True)
    
    pickle.dump(parameter_recovery_config_dict, 
                open(save_folder + '/' + model_name + '_parameter_recovery_run_config.pickle', 'wb'))
    
    print('Saving to: ')
    print(parameter_recovery_config_dict['save_file'])
    
    return parameter_recovery_config_dict

def make_parameter_recovery_dataset(model = 'angle',
                                    save_file = '',
                                    save_folder = '',
                                    n_subjects = 10,
                                    n_trials_per_subject = 1000,
                                    n_datasets = 100):
    
    data_dict = {}
    for i in range(n_datasets):
        # MAKE DATA
        data, parameter_dict = simulator_h_c(data = None,
                                             n_subjects = n_subjects,
                                             n_trials_per_subject = n_trials_per_subject,
                                             model = model,
                                             p_outlier = 0.0,
                                             conditions = None,
                                             depends_on = None,
                                             regression_models = None,
                                             regression_covariates = None,
                                             group_only_regressors = True,
                                             group_only = None,
                                             fixed_at_default = None)

        data_dict[i] = {'data': data,
                        'parameter_dict': parameter_dict}
        
        if not i % 10:
            print(str(i), ' of ', str(n_datasets), ' finished')
    
    # Create save_folder if not already there
    lanfactory.utils.try_gen_folder(folder = save_folder, 
                                    allow_abs_path_folder_generation = True)
    
    if save_file != '':
        print('Saving...')
        pickle.dump(data_dict, open(save_folder + save_file, 'wb'))
        print('Saved to ', save_folder + save_file)
        
    return data_dict
        


        
        
        
        
        
        
        

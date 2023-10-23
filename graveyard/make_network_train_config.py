# Append system path to include the config scripts
import sys
import os
from copy import deepcopy
import pickle

print('importing lanfactory')
import lanfactory

print('importing ssms')
import ssms

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config import *

import torch
import config
import pandas as pd
import yaml

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
    CLI.add_argument("--config_yaml",
                     type = none_or_str,
                     default = None)
    
    args = CLI.parse_args()
    print('Arguments passed: ', args)

    if args.config_yaml is None:
        ValueError('Need to pass a config yaml file path!')

    basic_config = yaml.safe_load(open('network_training_config_af.yaml', 'rb'))

    # Where do you want to save config files?
    network_train_config_save_folder = basic_config['PROJECT_FOLDER'] + '/data/config_files/network/' + \
                                        basic_config['NETWORK_TYPE'] + '/' + basic_config['MODEL'] + '/'


    # Specify training data folder:
    training_data_folder = basic_config['PROJECT_FOLDER'] + \
                            '/data/training_data/' + basic_config['GENERATOR_APPROACH'] + '/' + \
                                'training_data_n_samples' +  \
                                    '_' + str(N_SAMPLES) + '/' + basic_config['MODEL']

    # Specify the name of the config file
    network_train_config_save_name = 'train_config' + \
                                        '_opt_' + basic_config['OPTIMIZER_'] + \
                                            '_n_' + str(N_SAMPLES) + \
                                                '_dt_' + str(DELTA_T) + \
                                                    '_nps_' + str(N_PARAMETER_SETS) + \
                                                        '_npts_' + str(N_TRAINING_SAMPLES_BY_PARAMETER_SET) + \
                                                            '_architecture_search.pickle'

    # Train output type specifies what the network output node
    # 'represents' (e.g. log-probabilities / logprob, logits, probabilities / prob)

    # Specifically for cpn, we train on logit outputs for numerical stability, then transform outputs
    # to log-probabilities when running the model in evaluation / inference mode 
    train_output_type_dict = {'lan': 'logprob',
                            'cpn': 'logits',
                            'cpn_bce': 'prob'}

    # Last layer activation depending on train output type
    output_layer_dict = {'logits': 'linear',
                        'logprob': 'linear',
                        'prob': 'sigmoid'}

    # LOSS 
    # 'bce' (for binary-cross-entropy), use when train output is 'prob'
    # 'bcelogit' (for binary-cross-entropy with inputs representing logits) use when train output type is 'logits', (this is standard for cpns)
    # 'huber' (usually) used when train output is 'logprob'

    train_loss_dict = {'logprob': 'huber',
                    'logits': 'bcelogit',
                    'prob': 'bce'}

    data_key_dict = {'lan': {'features_key': 'data', 
                            'label_key': 'labels'},
                    'cpn': {'features_key': 'thetas',
                            'label_key': 'choice_p'}
                    }

    # Network architectures
    layer_sizes = basic_config['layer_sizes']
    activations = basic_config['activations']

    # Append last layer (type of layer depends on type of network as per train_output_type_dict dictionary above)
    activations = [act_tmp.append(output_layer_dict[train_output_type_dict[basic_config['NETWORK_TYPE']]]) for act_tmp in activations]

    weight_decays = basic_config['weight_decays'] # [0.0]

    # Train / validations split
    train_val_split = basic_config['train_val_split']
    # train_val_split = [0.98, 0.98, 0.98,
    #                    0.98, 0.98, 0.98]

    # Number is set to 10000 here (an upper bound), for training on all available data (usually roughly 300 files, but has never been more than 1000)
    # For numerical experiments, one may want to artificially constraint the number of training files to teest the impact on network performance
    n_training_files = basic_config['n_training_files']

    network_arg_dict = {'train_output_type': train_output_type_dict[basic_config['NETWORK_TYPE']],
                        'network_type': basic_config['NETWORK_TYPE']}

    # initial train_arg_dict
    # refined in for loop in next cell
    train_arg_dict_new = {'n_epochs': basic_config['N_EPOCHS'],
                        'loss': train_loss_dict[train_output_type_dict[basic_config['NETWORK_TYPE']]],
                        'optimizer': basic_config['OPTIMIZER_'],
                        'train_output_type': train_output_type_dict[basic_config['NETWORK_TYPE']],
                        # 'n_training_files': n_training_files[j],
                        # 'train_val_split': train_val_split[i],
                        'cpu_batch_size': basic_config['CPU_BATCH_SIZE'],
                        'gpu_batch_size': basic_config['GPU_BATCH_SIZE'],
                        'shuffle_files': basic_config['SHUFFLE'],
                        'label_lower_bound': eval(basic_config['labels_lower_bound']),
                        # 'weight_decay': weight_decays[k],
                        'learning_rate': basic_config['learning_rate'],
                        'features_key': data_key_dict[basic_config['NETWORK_TYPE']]['features_key'],
                        'label_key': data_key_dict[basic_config['NETWORK_TYPE']]['label_key'],
                        'save_history': True,
                        'lr_scheduler': basic_config['lr_scheduler'],
                        'lr_scheduler_params': basic_config['lr_scheduler_params']
                        }

    # Loop objects
    config_dict = {}
    network_arg_dicts = {}
    train_arg_dicts = {}

    cnt = 0
    for k in range(len(weight_decays)):
        for i in range(len(layer_sizes)):
            for j in range(len(n_training_files)):
                
                # Specify the arguments which you want to adjust in the network and train configs
                # For details check: lanfactory.config.network_config_mlp
                #                    lanfactory.config.train_config_mlp

                network_arg_dict['layer_sizes'] = layer_sizes[i]
                network_arg_dict['activations'] = activations[i]
                
                train_arg_dict['n_training_files'] = n_training_files[j]
                train_arg_dict['train_val_split'] = train_val_split[i]
                train_arg_dict['weight_decay'] = weight_decays[k]

                config_dict[cnt] = make_train_network_configs(training_data_folder=training_data_folder,
                                                              save_folder = network_train_config_save_folder,
                                                              train_val_split=train_val_split[i],
                                                              network_arg_dict = deepcopy(network_arg_dict),
                                                              train_arg_dict = deepcopy(train_arg_dict),
                                                              save_name = None)

                cnt += 1

    print('In total, ',
            len(list(config_dict.keys())),
                ' different networks will be trained with this config file')

    print('Now saving')

    # Create save_folder if not already there
    lanfactory.utils.try_gen_folder(folder = network_train_config_save_folder,
                                    allow_abs_path_folder_generation = True)
                    
    pickle.dump(config_dict, 
                open(network_train_config_save_folder + network_train_config_save_name, 'wb'))
    print(network_train_config_save_folder + network_train_config_save_name)    
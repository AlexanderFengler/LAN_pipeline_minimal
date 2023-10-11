import pickle
import numpy as np
import pandas as pd
import ssms
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
    CLI.add_argument("--config_file",
                     type = none_or_str,
                     default = None)
    CLI.add_argument('--config_dict_key',
                     type = none_or_int,
                     default = None)
    
    args = CLI.parse_args()
    print(args)
    
    assert args.config_file is not None, 'You need to supply a config file path to the script'
    
    if args.config_dict_key is not None:
        config = pickle.load(open(args.config_file, 'rb'))[args_config_dict_key]
    else:
        config = pickle.load(open(args.config_file, 'rb'))
        
    print('Printing config specs: ')
    print('GENERATOR CONFIG')
    print(config['data_config'])
          
    print('MODEL CONFIG')
    print(config['model_config'])
    
    # Make the generator
    print('Now generating data')
    my_dataset_generator = ssms.dataset_generators.lan_mlp.data_generator(generator_config = config['data_config'],
                                                                          model_config = config['model_config'])
    if 'cpn_only' in config['data_config'].keys():
        if config['data_config']['cpn_only']:
            x = my_dataset_generator.generate_data_training_uniform(save = True, cpn_only = True)
        else:
            x = my_dataset_generator.generate_data_training_uniform(save = True, cpn_only = False)
    else: # This is here for compatibility with legacy pipeline
        x = my_dataset_generator.generate_data_training_uniform(save = True)

    print('Data generation finished')
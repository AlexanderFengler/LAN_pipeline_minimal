# Append system path to include the config scripts
import sys
import os
import pickle
from copy import deepcopy
import argparse

print('importing lanfactory')
import lanfactory

print('importing ssms')
import ssms
import pandas as pd

def make_network_df(path = None,
                    model = None):
    assert path is not None, 'Please supply path to make_network_df'

    model_wise_pds = []
    pds = {}

    # Initial dictionary
    # with model_id as keys
    for file_ in os.listdir(path):
        if 'training_history' in file_:
            # print(file_)
            pd_tmp = pd.read_csv(path + '/' + file_)
            pd_tmp['model_id'] = file_[:file_.find('_')]
            pd_tmp['filename'] = file_
            pd_tmp['path'] = path + '/' + file_
            pd_tmp['model_type'] = model
            pds[file_[:file_.find('_')]] = pd_tmp
    
    #print(pds)
    # Extend torch model wise dictionary 
    # to include 'n_hidden_layers', 'size_hidden_layers'
    for m_id in pds.keys():
        for file_ in os.listdir(path):
            if m_id in file_ and 'network_config' in file_:
                network_config_tmp = pickle.load(open(path + '/' + file_, 'rb'))
                pds[m_id]['n_hidden_layers'] = len(network_config_tmp['layer_sizes']) - 1
                pds[m_id]['size_hidden_layers'] = network_config_tmp['layer_sizes'][0]
                
                if 'train_output_type' in network_config_tmp.keys():
                    pds[m_id]['train_output_type'] = network_config_tmp['train_output_type']
                else:
                    pds[m_id]['train_output_type'] = 'not supplied'
            if m_id in file_ and 'train_config' in file_:
                train_config_tmp = pickle.load(open(path + '/' + file_, 'rb'))
                
                if 'weight_decay' in train_config_tmp.keys():
                    pds[m_id]['weight_decay'] = train_config_tmp['weight_decay']
                else:
                    pds[m_id]['weight_decay'] = 'not supplied'
            
    # Turn the model_id wise dictionary into dataframe

    training_dat = pd.concat([pds[m_id] for m_id in pds.keys()]).reset_index(drop = True)
    print(pds.keys())
    print(training_dat)
    # Pick only the best models for a given configuration
    # of 'n_hidden_layers', 'size_hidden_layers' etc.
    best_models_pds = []
    for n_h_l in training_dat['n_hidden_layers'].unique():
        for s_h_l in training_dat['size_hidden_layers'].unique():
            training_dat_sub = training_dat.loc[(training_dat['n_hidden_layers'] == n_h_l) & \
                                                (training_dat['size_hidden_layers'] == s_h_l) & \
                                                (training_dat['epoch'] == training_dat['epoch'].max()), :]
            if training_dat_sub.shape[0] > 0:
                val_loss_min = training_dat_sub['val_loss'].min()
                best_models_pds.append(training_dat_sub.loc[training_dat_sub['val_loss'] == val_loss_min, :])

    # Turn best_models dictionary into dataframe across architectures
    best_models_dat = pd.concat(best_models_pds)

    # Append dataframe to list of model_wise dataframes 
    model_wise_pds.append(best_models_dat)
    full_dat = pd.concat(model_wise_pds).reset_index(drop = True).drop(['Unnamed: 0'], axis = 1)

    # Pick only the rows concerning final epochs
    training_dat = training_dat.loc[training_dat['epoch'] == training_dat['epoch'].max(), :]
    return full_dat, training_dat
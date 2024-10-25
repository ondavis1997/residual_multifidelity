#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ondavis
"""

import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from networks import *
from train_test_validate import *
from datasets_dataloaders import *
from tqdm import tqdm
import pickle
from early_stop import *


def initialize_weights(m):

  if isinstance(m, torch.nn.Linear):
      torch.nn.init.kaiming_uniform_(m.weight.data)
      torch.nn.init.uniform_(m.bias.data, 0.0, 1.0)
      
def enable_dropout(model,tol):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            m.p = tol

def disable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()
            
def convert_tensor_1D(X):
    X_tensor = torch.zeros(X.shape[0],1)
    for i in range(X.shape[0]) :
        X_tensor[i,0] = X[i]
    return X_tensor

def convert_tensor_2D(X):
    X_tensor = torch.zeros(X.shape[0],X.shape[1])
    for i in range(X.shape[0]) :
        for j in range(X.shape[1]) :
            X_tensor[i,j] = X[i,j]
    return X_tensor

def convert_np_1D(X):
    X_np = np.zeros(X.shape[0])
    for i in range(X.shape[0]) :
        X_np[i] = X[i,0]
    return X_np

def convert_np_2D(X):
    X_np = np.zeros(X.shape[0],X.shape[1])
    for i in range(X.shape[0]) :
        for j in range(X.shape[1]) :
            X_np[i][j] = X[i,j]
    return X_np



width_depth = [(10,2)]
sample_numbers = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]# 45, 50, 55, 60, 65, 70, 75, 80]
batch_sizes = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
params = list(zip(sample_numbers,batch_sizes))
seeds = [53, 101, 151, 203, 257, 305, 343, 413, 467, 501, 539, 695, 721, 799, 831, 893, 915, 953, 1033, 1137]

'''
width_depth = [(25,7)]
sample_numbers = [5000, 7000, 9000, 11000, 13000, 15000, 17000]
batch_sizes = [128, 128, 128, 128, 128, 128, 128]
params = list(zip(sample_numbers,batch_sizes))
seeds = [53, 101, 151, 203, 257, 305, 343, 413, 467, 501, 539, 695, 721, 799, 831, 893, 915, 953, 1033, 1137]
'''

'''
width_depth = [(7,15)]
sample_numbers = [1000,2000,3000,5000,7000,9000,11000,13000,15000,17000]
batch_sizes = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
params = list(zip(sample_numbers,batch_sizes))
seeds = [53, 101, 151, 203, 257, 343, 413, 539, 695, 721, 799, 831, 893, 915, 953, 1033, 1137]
'''

ensemble = {} 

for w, d in width_depth:
    
    print(f'working on width ={w}, depth ={d}')
    num_blocks = int((d-3)/2)
    model = DenseResnet(in_features=1, out_features=1, num_res_blocks = num_blocks, width = w)
    error_data = np.zeros((len(sample_numbers),4))
        
        
    for j, (num_samples, b_size) in enumerate(params):
        
        print(f'working on {num_samples} samples')
        
        # Load training, testing, and validation sets
        train_data = DataLoader(TrainData(f'../training_data/RLNN_training_data_{num_samples}.txt'), batch_size = b_size, shuffle=False)
        test_data = np.loadtxt(f'../training_data/RLNN_test_data_{num_samples}.txt',dtype=float)
        val_data = np.loadtxt(f'../training_data/RLNN_validation_data_{num_samples}.txt', dtype=float)
        test_pred = np.loadtxt(f'../training_data/RLNN_test_pred_{num_samples}.txt', dtype =  float)
        val_pred = np.loadtxt(f'../training_data/RLNN_val_pred_{num_samples}.txt', dtype = float)
        LF_pred_on_test = np.loadtxt(f'../training_data/RLNN_LF_pred_on_test_inputs_{num_samples}.txt',dtype=float)
        LF_pred_on_val = np.loadtxt(f'../training_data/RLNN_LF_pred_on_validation_inputs_{num_samples}.txt',dtype=float)
        # Split up test/validation sets into inputs and outputs and convert to torch tensors

        x_test = test_data[:,:-1]
        y_test = test_data[:,-1]
        x_test_torch = convert_tensor_2D(x_test)
        y_test_torch = convert_tensor_1D(y_test)

        x_val = val_data[:,:-1]
        y_val = val_data[:,-1]
        x_val_torch = convert_tensor_2D(x_val)
        y_val_torch = convert_tensor_1D(y_val)

        val_pred_torch = convert_tensor_1D(val_pred)
        test_pred_torch = convert_tensor_1D(test_pred)
        
        ensemble_L2 =  np.zeros(len(seeds),)
        ensemble_max = np.zeros(len(seeds),)
        
        for trial, seed in enumerate(seeds):
            print(f'trial {trial} with seed {seed}')
            torch.manual_seed(seed)
            # Set training parameters 
            learning_rate = 1e-2
            reg_weight = 1e-3
            
            model.apply(initialize_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_weight)
            loss_fn = torch.nn.MSELoss(reduction ='mean')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
            factor=0.5, patience=10, threshold=1e-5, threshold_mode='abs', verbose=False)
            stop_criteria = EarlyStop(patience = 500, delta = 0.00000025)
  
            # Set number of epochs
    
            EPOCHS = 3000
            
            train_errs = []
            val_errs = []
            
                
            for i in tqdm(range(EPOCHS)):
                loss = train_loop(train_data, model, loss_fn, optimizer)
                train_errs.append(loss)
                
                with torch.no_grad():
                    
                    pred = model(x_val_torch).float()
                    val_loss = loss_fn(pred,val_pred_torch)
                    
                val_errs.append(val_loss.item())
                
                if i%50 == 0:
                    print(val_loss.item())
                
                if stop_criteria.terminate_training(val_loss.item()) and  i > 150:
                    break
                    
                scheduler.step(val_loss.item())
             
            
            '''
            plt.plot(np.arange(len(train_errs)), train_errs, label = 'train error')
            plt.plot(np.arange(len(val_errs)), val_errs, label = 'validation error')
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
            plt.title('monitor for overfitting')
            plt.show()
            plt.close()
            '''
    
            
            model.eval()
            
            y_test_surr = model(x_test_torch).float()
            y_test_surr_np = convert_np_1D(y_test_surr)
            y_test_results_np = y_test_surr_np + LF_pred_on_test
            y_test_results_torch = convert_tensor_1D(y_test_results_np)
            test_loss = loss_fn(y_test_results_torch, y_test_torch)
            maxerr = np.max(abs(y_test-y_test_results_np))
            
            ensemble_L2[trial] = test_loss.item()
            ensemble_max[trial] = maxerr
        
        ensemble[f'{num_samples}'] = ensemble_L2
        L2_mean = np.mean(ensemble_L2)
        max_mean = np.mean(ensemble_max)
        std_dev = np.std(ensemble_L2)
        
        print(f'  max error = {max_mean}')
        print(f'  L2 error = {L2_mean}')
        
        error_data[j,0] = num_samples
        error_data[j,1] = max_mean
        error_data[j,2] = L2_mean
        error_data[j,3] = std_dev  
    

    pickle.dump(ensemble, open(f'../post_process_and_plots/ensemble_DiscrepNN_width_{w}_depth_{d}.pkl','wb'))





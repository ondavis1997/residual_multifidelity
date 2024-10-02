#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 20:56:42 2022

@author: owendavis
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
from early_stop import *
import pickle

def initialize_weights(m):

  if isinstance(m, torch.nn.Linear):
      torch.nn.init.kaiming_uniform_(m.weight.data)
      torch.nn.init.uniform_(m.bias.data, 0.0, 1.0)
      
            
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


width_depth = [(7,7)]
sample_numbers = [250,500,1000,2000,3000,5000,7000,9000,11000,13000,15000,17000]
batch_sizes = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
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
    model = DenseResnet(in_features=4, out_features=1, num_res_blocks = num_blocks, width = w)
    error_data = np.zeros((len(sample_numbers),4))
        
    for j, (num_samples, b_size) in enumerate(params):
        
        print(f'working on {num_samples} samples')
        
        # Load training, testing, and validation sets
        train_data = DataLoader(TrainData(f'../training_data/HF_ONLY_training_data_{num_samples}.txt'), batch_size = 128, shuffle=True)
        test_data = np.loadtxt(f'../training_data/SF_test_data_{num_samples}.txt',dtype=float)
        val_data = np.loadtxt(f'../training_data/SF_validation_data_{num_samples}.txt', dtype=float)

        # Split up test/validation sets into inputs and outputs and convert to torch tensors

        x_test = test_data[:,:-1]
        y_test = test_data[:,-1]
        x_test_torch = convert_tensor_2D(x_test)
        y_test_torch = convert_tensor_1D(y_test)

        x_val = val_data[:,:-1]
        y_val = val_data[:,-1]
        x_val_torch = convert_tensor_2D(x_val)
        y_val_torch = convert_tensor_1D(y_val)
        
        ensemble_L2 =  np.zeros(len(seeds),)
        ensemble_max = np.zeros(len(seeds),)

        for trial, seed in enumerate(seeds):
            
            print(f' trial {trial} with seed {seed}')
            
            torch.manual_seed(seed)
            # Set training parameters 
            learning_rate = 1e-2
            reg_weight = 1e-5
            
            model.apply(initialize_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_weight)
            loss_fn = torch.nn.MSELoss(reduction ='mean')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
            factor=0.5, patience=10, threshold=0.0001, threshold_mode='abs', verbose=False)
            stop_criteria = EarlyStop(patience = 100 , delta = 0.00005)
    
            # Set number of epochs
    
            EPOCHS = 3000
            
            train_errs = []
            val_errs = []
            
    
                
            
            for i in tqdm(range(EPOCHS)):
                loss = train_loop(train_data, model, loss_fn, optimizer)
                train_errs.append(loss)
                
                with torch.no_grad():
                    
                    pred = model(x_val_torch).float()
                    val_loss = loss_fn(pred,y_val_torch)
                    pred_np = convert_np_1D(pred)
                    
                
                val_errs.append(val_loss.item())
                
                if i%50 == 0:
                    print(val_loss.item())
                
                if stop_criteria.terminate_training(val_loss.item()) and  i > 150:
                    break
                    
                scheduler.step(val_loss.item())
             
            
            
            model.eval()
        
     
            
            y_test_surr = model(x_test_torch).float()
            test_loss = loss_fn(y_test_surr, y_test_torch)
            y_test_surr_np = convert_np_1D(y_test_surr)
            maxerr = np.max(abs(y_test-y_test_surr_np))
            
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


    pickle.dump(ensemble, open(f'ensemble_HFNN_width_{w}_depth_{d}.pkl', 'wb'))


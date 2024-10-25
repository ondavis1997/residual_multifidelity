#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: owendavis
"""

import numpy as np
from test_function import f, f_LF
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import os


equivs = [4,8,12,16,20,24, 28, 32, 36, 40, 44, 48, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80] #250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000]
HF  = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80] #250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000]
LF  = [4, 8, 12,16,20, 24, 28, 32, 36, 40, 44, 48, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]# 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000]
num_test = 5000
num_val = 5000
'''
equivs = [5000]
HF = [5000]
LF = [5000]
num_test = 5000
num_val = 5000
'''
for equivs, num_HF, num_LF in list(zip(equivs,HF,LF)):
    
    total = equivs + num_test + num_val
     
    x = np.random.uniform(0,1,total).reshape(total,1)
    inputs = list(x)
    
    inputs_test = inputs[:num_test]
    inputs_val = inputs[num_test:num_test + num_val]
    inputs_shared = inputs[num_test+num_val:]
    inputs_res = np.zeros((num_HF,2))
    


    
    outputs_res = np.zeros(num_HF,)
    outputs_HF = np.zeros(num_HF,)
    outputs_LF = np.zeros(num_HF,)
    LF_pred_on_test = np.zeros(num_test,)
    HF_pred_on_test = np.zeros(num_test,)
    LF_pred_on_val = np.zeros(num_val,)
    HF_pred_on_val = np.zeros(num_val)
    
    for i in range(num_test):
        HF_pred_on_test[i] = f(*inputs_test[i])
        LF_pred_on_test[i] = f_LF(*inputs_test[i])
        
    for i in range(num_val):
        HF_pred_on_val[i] = f(*inputs_val[i])
        LF_pred_on_val[i] = f_LF(*inputs_val[i])
        
    for i in range(equivs):
        outputs_LF[i] = f_LF(*inputs_shared[i])
        outputs_HF[i] = f(*inputs_shared[i])
        outputs_res[i] = f(*inputs_shared[i]) - f_LF(*inputs_shared[i])
        
        
    validate_data_MF = np.zeros((num_val, 3))
    validate_data_SF = np.zeros((num_val, 2))
    test_data_MF = np.zeros((num_test, 3))
    test_data_SF = np.zeros((num_test,2))
    train_data_res = np.zeros((num_HF, 3))
    train_data_HF = np.zeros((num_HF,2))
    train_data_DNL = np.zeros((num_HF, 3))
    
    
    validate_data_MF[:,:-2] = inputs_val
    validate_data_MF[:,-2] = LF_pred_on_val
    validate_data_MF[:,-1] = HF_pred_on_val
    
    validate_data_SF[:,:-1] = inputs_val
    validate_data_SF[:,-1] = HF_pred_on_val
    
    test_data_MF[:,:-2] = inputs_test
    test_data_MF[:,-2] = LF_pred_on_test
    test_data_MF[:,-1] = HF_pred_on_test
    
    test_data_SF[:,:-1] = inputs_test
    test_data_SF[:,-1] = HF_pred_on_test
    
    train_data_res[:,:-2] = inputs_shared
    train_data_res[:,-2] = outputs_LF
    train_data_res[:,-1] = outputs_res
    
    train_data_HF[:,:-1] = inputs_shared
    train_data_HF[:,-1] = outputs_HF
    
    train_data_DNL[:,:-2] = inputs_shared
    train_data_DNL[:,-2] = outputs_LF
    train_data_DNL[:,-1] = outputs_HF
    
    RMFNN_test_outputs = HF_pred_on_test - LF_pred_on_test
    RMFNN_val_outputs = HF_pred_on_val - LF_pred_on_val
    
    
    scalar = MinMaxScaler()
    
    # Rescale MF inputs
    x_train_MF = train_data_res[:,:-1]  
    x_test_MF = test_data_MF[:,:-1]
    x_validate_MF = validate_data_MF[:,:-1]
    
    x_train_MF = scalar.fit_transform(x_train_MF)
    x_test_MF = scalar.transform(x_test_MF)
    x_validate_MF = scalar.transform(x_validate_MF)
    
    train_data_res[:,:-1] = x_train_MF
    train_data_DNL[:,:-1] = x_train_MF
    test_data_MF[:,:-1] = x_test_MF
    validate_data_MF[:,:-1] = x_validate_MF

    
    # Rescale SF inputs
    x_train_SF = train_data_HF[:,:-1]
    x_test_SF = test_data_SF[:,:-1]
    x_validate_SF = validate_data_SF[:,:-1]
     
    x_train_SF = scalar.fit_transform(x_train_SF)
    x_test_SF = scalar.transform(x_test_SF)
    x_validate_SF = scalar.transform(x_validate_SF)
    
    train_data_HF[:,:-1] = x_train_SF
    test_data_SF[:,:-1] = x_test_SF
    validate_data_SF[:,:-1] = x_validate_SF


    # Scaled version of HF function which is has reduction in uniform norm by factor of 100

    alpha = 1/10 #scaling constant
    alpha_string = 10  

    validate_data_SF_scaled = np.zeros_like(validate_data_SF)
    validate_data_SF_scaled[:,:-1] = validate_data_SF[:,:-1]
    validate_data_SF_scaled[:,-1] = alpha*validate_data_SF[:,-1]

    test_data_SF_scaled = np.zeros_like(test_data_SF)
    test_data_SF_scaled[:,:-1] = test_data_SF[:,:-1]
    test_data_SF_scaled[:,-1] = alpha*test_data_SF[:,-1]

    train_data_HF_scaled = np.zeros_like(train_data_HF)
    train_data_HF_scaled[:,:-1] = train_data_HF[:,:-1]
    train_data_HF_scaled[:,-1] = alpha*train_data_HF[:,-1]

    
    
        
    # Save Training Data
    np.savetxt(f'training_data/RMFNN_training_data_{equivs}.txt', train_data_res)
    np.savetxt(f'training_data/DNLNN_training_data_{equivs}.txt',train_data_DNL)
    np.savetxt(f'training_data/HF_ONLY_training_data_{equivs}.txt', train_data_HF)
    
    # Save Testing Data
    np.savetxt(f'training_data/MF_test_data_{equivs}.txt', test_data_MF)
    np.savetxt(f'training_data/SF_test_data_{equivs}.txt', test_data_SF)
    np.savetxt(f'training_data/RMFNN_LF_pred_on_test_inputs_{equivs}.txt', LF_pred_on_test)
    np.savetxt(f'training_data/RMFNN_test_pred_{equivs}.txt', RMFNN_test_outputs)
    
    # Save Validation Data
    np.savetxt(f'training_data/MF_validation_data_{equivs}.txt', validate_data_MF)
    np.savetxt(f'training_data/SF_validation_data_{equivs}.txt', validate_data_SF)
    np.savetxt(f'training_data/RMFNN_LF_pred_on_validation_inputs_{equivs}.txt', LF_pred_on_val)
    np.savetxt(f'training_data/RMFNN_val_pred_{equivs}.txt', RMFNN_val_outputs)

    
    
        
        

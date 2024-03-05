#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 08:36:38 2024

@author: owendavis
"""

import numpy as np

import matplotlib.pyplot as plt
import os

#os.chdir('RMFNN_codes_and_data')


# load  data


width = 7
depth = 7

HF_res = np.loadtxt(f'plot_data_HF_resnet_L2_width_{width}_depth_{depth}.txt', dtype=float) 
RMFNN_res = np.loadtxt(f'plot_data_RMFNN_resnet_L2_width_{width}_depth_{depth}.txt', dtype=float) 
MFNN_res = np.loadtxt(f'plot_data_MFNN_resnet_L2_width_{width}_depth_{depth}.txt', dtype=float)

num_samples = [500,1000,2000, 3000,5000,7000,9000,11000,13000,15000,17000] 


HF_res_expected_errors = HF_res[:,2]
HF_res_std_dev = HF_res[:,3]


RMFNN_res_expected_errors = RMFNN_res[:,2]
RMFNN_res_std_dev = RMFNN_res[:,3]


MFNN_res_expected_errors = MFNN_res[:,2]
MFNNres_std_dev = MFNN_res[:,3]


plt.scatter(num_samples[:], HF_res_expected_errors, color = 'b', label = 'HFNN ResNet',marker='^',s=20)
plt.scatter(num_samples[:], MFNN_res_expected_errors, color = 'g', label = 'MFNN ResNet', marker = 'x', s=20)
plt.scatter(num_samples[:], RMFNN_res_expected_errors, color = 'r', label = 'RMFNN ResNet', s=20)


plt.xlabel('Number of HF Samples', fontsize=14)
plt.ylabel(r'$MSE$', fontsize = 18)
plt.yscale('log')
plt.legend()
#plt.savefig('OND_numerical_example.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()
plt.close()
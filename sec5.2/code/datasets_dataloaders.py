#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: owendavis
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from networks import *



class TrainData(Dataset):
    def __init__(self, source_file):
        
        self.raw_data_np = np.loadtxt(source_file, dtype=float)
        self.x_train_np = self.raw_data_np[:,:-1]
        self.y_train_np = self.raw_data_np[:,-1]
        
        self.x_train = torch.from_numpy(self.x_train_np).float()
        self.y_train = torch.from_numpy(self.y_train_np).float()
        
    def __len__(self):
        return len(self.raw_data_np)
    
    def __getitem__(self, index):
        
        x_train = self.x_train[index,:].float()
        y_train = self.y_train[index].float()
        
        return x_train, y_train
    
class TestData(Dataset):
    def __init__(self, source_file):
        
        self.raw_data_np = np.loadtxt(source_file, dtype=float)
        self.x_test_np = self.raw_data_np[:,:-1]
        self.y_test_np = self.raw_data_np[:,-1]
        
        self.x_test = torch.from_numpy(self.x_test_np).float()
        self.y_test = torch.from_numpy(self.y_test_np).float()
        
    def __len__(self):
        return len(self.raw_data_np)
    
    def __getitem__(self,index):
        x_test = self.x_test[index,:].float()
        y_test = self.y_test[index].float()
        
        return x_test, y_test
    
    
    
        
    
    






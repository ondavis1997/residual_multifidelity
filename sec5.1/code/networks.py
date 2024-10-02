#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:54:52 2022

@author: owendavis

"""

import torch
from torchinfo import summary



#==========================================================================================
# Dense Resnets (with and without dropout)
#==========================================================================================

class ResBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        
        self.activate1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.width, self.width)
        self.activate2 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.width, self.width)
       
    
    def forward(self, x):
        res = x
        x = self.activate1(x)
        x = self.linear1(x)
        x = self.activate2(x)
        x = self.linear2(x)
        x += res
        return x
    
    
    
    
class DenseResnet(torch.nn.Module):
    def __init__(self, in_features, out_features, num_res_blocks, width):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.width = width
        self.num_res_blocks = num_res_blocks
        
        self.linear_input = torch.nn.Linear(self.in_features, self.width)
        self.activate1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.width, self.width)
        self.activate2 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.width, self.width)
        self.blocks = torch.nn.Sequential(*[ResBlock(self.width) for _ in range(self.num_res_blocks)])
        self.activate_post_res = torch.nn.ReLU()
        self.linear_output = torch.nn.Linear(self.width, self.out_features)
        
    def forward(self,x):
        x = self.linear_input(x)
        x = self.activate1(x)
        x = self.linear1(x)
        x = self.blocks(x)
        x = self.activate_post_res(x)
        x = self.linear_output(x)
        return x
    

class ResDropoutBlock(torch.nn.Module):
    def __init__(self, width, droptol):
        super().__init__()
        self.width = width
        self.droptol = droptol
        
        self.activate1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.width, self.width)
        self.drop1 = torch.nn.Dropout(self.droptol)
        self.activate2 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.width, self.width)
        self.drop2 = torch.nn.Dropout(self.droptol)
       
    
    def forward(self, x):
        res = x
        x = self.activate1(x)
        x = self.linear1(x)
        x = self.drop1(x)
        x = self.activate2(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x += res
        return x
    
class DenseDropoutResnet(torch.nn.Module):
    def __init__(self, in_features, out_features, num_res_dropout_blocks, width, droptol = 0.05):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.width = width
        self.num_res_dropout_blocks = num_res_dropout_blocks
        self.droptol = droptol
        
        self.linear_input = torch.nn.Linear(self.in_features, self.width)
        self.drop1 = torch.nn.Dropout(self.droptol)
        self.activate1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.width, self.width)
        self.drop2 = torch.nn.Dropout(self.droptol)
        self.blocks = torch.nn.Sequential(*[ResDropoutBlock(self.width, self.droptol) for _ in range(self.num_res_dropout_blocks)])
        self.activate_post_res = torch.nn.ReLU()
        self.linear_output = torch.nn.Linear(self.width, self.out_features)
        
    def forward(self,x):
        x = self.linear_input(x)
        x = self.drop1(x)
        x = self.activate1(x)
        x = self.linear1(x)
        x = self.drop2(x)
        x = self.blocks(x)
        x = self.activate_post_res(x)
        x = self.linear_output(x)
        return x
    

#==========================================================================================
# Standard Networks (with and without dropout)
#==========================================================================================
    
    
    

class BasicBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        
        self.width = width
        self.activate1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.width,self.width)
        self.activate2 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.width, self.width)

    def forward(self,x):
        x = self.activate1(x)
        x = self.linear1(x)
        x = self.activate2(x)
        x = self.linear2(x)
        
        return x
    
class StandardNet(torch.nn.Module):
    def __init__(self, in_features, out_features, num_basic_blocks, width):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.width = width
        self.num_basic_blocks = num_basic_blocks
        self.linear_input = torch.nn.Linear(self.in_features, self.width)
        self.activate1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.width, self.width)
        self.activate2 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.width, self.width)
        self.blocks = torch.nn.Sequential(*[BasicBlock(self.width) for _ in range(self.num_basic_blocks)])
        self.activate_post_blocks = torch.nn.ReLU()
        self.linear_output = torch.nn.Linear(self.width, self.out_features)
        
    def forward(self,x):
        x = self.linear_input(x)
        x = self.activate1(x)
        x = self.linear1(x)
        x = self.blocks(x)
        x = self.activate_post_blocks(x)
        x = self.linear_output(x)
        return x
    


class BasicDropoutBlock(torch.nn.Module):
    def __init__(self, width, droptol):
        super().__init__()
        
        self.width = width
        self.droptol = droptol
        self.activate1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.width,self.width)
        self.drop1 = torch.nn.Dropout(self.droptol)
        self.activate2 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.width, self.width)
        self.drop2 = torch.nn.Dropout(self.droptol)

    def forward(self,x):
        x = self.activate1(x)
        x = self.linear1(x)
        x = self.drop1(x)
        x = self.activate2(x)
        x = self.linear2(x)
        x = self.drop2(x)
        
        return x
    
class StandardDropoutNet(torch.nn.Module):
    def __init__(self, in_features, out_features, num_basic_dropout_blocks, width, droptol=0.5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.width = width
        self.num_basic_dropout_blocks = num_basic_dropout_blocks
        self.droptol = droptol
        self.linear_input = torch.nn.Linear(self.in_features, self.width)
        self.drop1 = torch.nn.Dropout(self.droptol)
        self.activate1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.width, self.width)
        self.drop2 = torch.nn.Dropout(self.droptol)
        self.blocks = torch.nn.Sequential(*[BasicDropoutBlock(self.width,self.droptol) for _ in range(self.num_basic_dropout_blocks)])
        self.activate_post_blocks = torch.nn.ReLU()
        self.linear_output = torch.nn.Linear(self.width, self.out_features)
        
    def forward(self,x):
        x = self.linear_input(x)
        x = self.drop1(x)
        x = self.activate1(x)
        x = self.linear1(x)
        x = self.drop2(x)
        x = self.blocks(x)
        x = self.activate_post_blocks(x)
        x = self.linear_output(x)
        return x
    

        

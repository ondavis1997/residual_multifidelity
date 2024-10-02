#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:40:48 2022

@author: owendavis

"""


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np




def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X).float()
        pred = pred.squeeze(-1)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''
        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        ''' 
    return loss.item()


def validation_loop(dataloader, model, loss_fn):
   size = len(dataloader.dataset)
   num_batches = len(dataloader)
   test_loss = 0

   with torch.no_grad():
       for X, y in dataloader:
           pred = model(X).float()
           test_loss += loss_fn(pred, y).item()
           

   test_loss /= num_batches
   print(f"Avg loss: {test_loss:>8f} \n")



def test_loop(dataloader, model, loss_fn):
   size = len(dataloader.dataset)
   num_batches = len(dataloader)
   test_loss = 0

   with torch.no_grad():
       for X, y in dataloader:
           pred = model(X).float()
           test_loss += loss_fn(pred, y).item()
           

   test_loss /= num_batches
   
   #print(f"Avg loss: {test_loss:>8f} \n")
   return test_loss

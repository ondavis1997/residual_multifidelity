
import numpy as np


class EarlyStop():
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_val_loss = np.inf
    
    def terminate_training(self, val_loss):
        if val_loss < (self.min_val_loss - self.delta):
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss - self.delta):
            self.counter +=1
            if self.counter > self.patience:
                return True
        return False
    
    def progress(self):
        print(self.counter)
        
        

"""
Autoencoder class implentation
"""
import torch.nn as nn

class Autoencoder(nn.Module):
    """ autoencoder class"""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(28*28,128),  # N,784 -> N,128
            nn.ReLU(), 
            nn.Linear(128,64),   # N,784 -> N,128
        )
    
    def forward(self, x):
        """
        
        """
        pass
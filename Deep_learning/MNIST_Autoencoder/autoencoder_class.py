"""
Autoencoder class implentation
"""
from base64 import encode
import torch.nn as nn

class Autoencoder(nn.Module):
    """ autoencoder class"""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(28*28,128),  # N,784 -> N,128
            nn.ReLU(), 
            nn.Linear(128,64),   # N,128 -> N,64
            nn.ReLU(),
            nn.Linear(64,12),   # N,64 -> N,12
            nn.ReLU(),
            nn.Linear(12,3),   # N,12 -> N,3
        )

        self.decoder=nn.Sequential(
            nn.Linear(3,12),    # N,3 -> N,12
            nn.ReLU(), 
            nn.Linear(12,64),   # N,12 -> N,64
            nn.ReLU(),
            nn.Linear(64,128),  # N,64 -> N,128
            nn.ReLU(),
            nn.Linear(128,784), # N,128 -> N,784 
            # Note: Sigmoid func converts values of last layer 
            # between 0 and 1
            nn.Sigmoid()    
        )

    def forward(self, x):
        """
        
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
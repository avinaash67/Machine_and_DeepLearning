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
            # Note: "nn. Linear(input layer size, output layer size)" 
            # creates the weight and bias matrices for that specific layer.
            # i.e. y = mx + b. Where, m = weight matrix; b= bias matrix
            # Weight matrix shape = input layer size x output layer size
            # Bias matrix shape = input layer size x 1
            nn.Linear(28*28,512),  # N,784 -> N,128
            nn.ReLU(), 
            nn.Linear(512,256),  # N,784 -> N,128
            nn.ReLU(),
            nn.Linear(256,256),  # N,784 -> N,128
            nn.ReLU(),
            nn.Linear(256,128),  # N,784 -> N,128
            nn.ReLU(),
            nn.Linear(128,128),   # N,128 -> N,64
            # nn.ReLU(),
            # nn.Linear(128,64),   # N,128 -> N,64
            # nn.ReLU(),
            # nn.Linear(64,12),   # N,64 -> N,12
            # nn.ReLU(),
            # nn.Linear(12,3),   # N,12 -> N,3

            # nn.Conv2d(1, 6, kernel_size=5),
            # nn.ReLU(True),
            # nn.Conv2d(6,16,kernel_size=5),
            # nn.ReLU(True)
        )

        self.decoder=nn.Sequential(
            # nn.Linear(3,12),    # N,3 -> N,12
            # nn.ReLU(), 
            # nn.Linear(12,64),   # N,12 -> N,64
            # nn.ReLU(),
            # nn.Linear(64,128),  # N,64 -> N,128
            # nn.ReLU(),
            nn.Linear(128,128),  # N,64 -> N,128
            nn.ReLU(),
            nn.Linear(128,256), # N,128 -> N,784 
            nn.ReLU(),
            nn.Linear(256,256), # N,128 -> N,784 
            nn.ReLU(),
            nn.Linear(256,512), # N,128 -> N,784
            nn.ReLU(),
            nn.Linear(512,784), # N,128 -> N,784
            # # Note: Sigmoid func converts values of last layer 
            # # between 0 and 1
            nn.Sigmoid()    

            # nn.ConvTranspose2d(16,6,kernel_size=5),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(6,1,kernel_size=5),
            # nn.ReLU(True),
            # nn.Sigmoid()
        )

    def forward(self, x):
        """Feed forward neural network function"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder_conv(nn.Module):
    """ """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Feed forward neural network function"""
        pass
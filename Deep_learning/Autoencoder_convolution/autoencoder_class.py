"""
Autoencoder class implentation using CNN
"""
from base64 import encode
import torch.nn as nn


class Autoencoder_conv(nn.Module):
    """  autoencoder class cnn"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        """Feed forward neural network function"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
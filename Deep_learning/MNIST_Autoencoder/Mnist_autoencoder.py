import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
from autoencoder_class import Autoencoder
import torch.nn as nn
import matplotlib.pyplot as plt


# Downloading MNIST Dataset and saving it
mnist_data = MNIST(root='../../data/MNIST',
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]))

# Dataloader object
data_loader = DataLoader(dataset=mnist_data,
                        batch_size=2,
                        #shuffle=True
                        )

# Iterator
dataiter = iter(data_loader)
# Runs through the batches of data
data=dataiter.next() 

# # Verification
# data_flat=data[0][1].view(28,28) # Flattens and plots image tensor
# print(data_flat[1][1]) # prints number
# plt.savefig("test_img.png")

# Autoencoder object
# model = Autoencoder()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
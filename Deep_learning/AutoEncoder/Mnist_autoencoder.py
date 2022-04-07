import torch

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader

# Downloading MNIST Dataset and saving it
mnist_data = MNIST(root='../../data/MNIST',
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]))

# Dataloader object
data_loader = DataLoader(dataset=mnist_data,
                        batch_size=1,
                        shuffle=True)

# Creating an iterator
dataiter = iter(data_loader)
img,label = dataiter.next()
print(img)
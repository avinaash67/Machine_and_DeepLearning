from os import device_encoding
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
from autoencoder_class import Autoencoder
import torch.nn as nn
import matplotlib.pyplot as plt
from autoencoder_functions import fit

BATCH_SIZE = 50

# Transforms
transform = transforms.Compose([transforms.ToTensor()])

# Train_dataset
train_dataset = MNIST(
    root="../../data/MNIST",train=True, transform=transform, download=True
)
# Test_dataset
test_dataset = MNIST(
    root="../../data/MNIST", train=False, transform=transform, download=True
)
# Train_dataloader
train_dl = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
# Test_dataloader
test_dl = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

# Iterator
dataiter = iter(train_dl)
# Runs through the batches of data
data=dataiter.next() 

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Autoencoder object
model = Autoencoder()
# load it to the specified device, either gpu or cpu
model.to(device=device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)

# Calling fit method for training
outputs = fit(epochs=10,model=model, criterion=criterion,optimizer=optimizer,
            train_dl=train_dl,test_dl=test_dl,metric=None)
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

dataset = MNIST('../../data/MNIST',
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]))
print('\n--> MNIST "training" dataset downloaded. Consists of', len(dataset), 'datasets\n')
img, label = dataset[0]
plt.imshow(img[0], cmap='gray')


# split_indices function is used to split dataset into "train and validation sets"
# Parameters
# n_val = size of validation dataset
# val_pct = percentage of validation dataset
def split_indices(n, val_pct):
    n_val = int(val_pct * n)
    # Create random permutation from 0 to n-1 from the entire dataset
    idxs = np.random.permutation(n)
    # Pick first set of n_val as validation dataset
    return idxs[n_val:], idxs[:n_val]


# Checking split_indices
train_indices, val_indices = split_indices(len(dataset), 0.2)

# Creating batches of data
batchsize = 100
# SubsetRandomSampler Class samples random samples from the given batch
# Parameter "sampler=train_sampler" denotes, Dataloader uses data
# from the train_sampler dataset
train_sampler = SubsetRandomSampler(train_indices)
train_dl = DataLoader(dataset=dataset,
                      batch_size=batchsize,
                      sampler=train_sampler)
# Parameter "sampler=val_sampler" denotes,
# Dataloader uses data from the val_sampler dataset
val_sampler = SubsetRandomSampler(val_indices)
val_dl = DataLoader(dataset=dataset,
                    batch_size=batchsize,
                    sampler=val_sampler)


# # Check --> Dataloader
# for xb, yb in train_dl:         # First batch is split into xb and yb
#     # print(xb[0])                # First element of first batch is printed
#     print(xb.shape)             # Shape of the tensor
#     print(xb.reshape(100, 784).shape)
#     break                       # Breaking "For" loop


class MnistModel(nn.Module):
    """Feedforward neural network with 1 hidden layer"""
    def __init__(self, in_size, hidden_size1, hidden_size2, out_size):
        super().__init__()
        # Input and Hidden layer1
        self.linear1 = nn.Linear(in_size, hidden_size1)
        # Hidden layer1 and Hidden layer2
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        # Hidden layer2 and output layer
        self.linear3 = nn.Linear(hidden_size2, out_size)

    def forward(self, xb):
        # Flattening image tensors
        xb = xb.view(xb.size(0), -1)  # xb.size(0) gives the batch size
        # Get intermediate outputs using hidden layer 1
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using hidden layer2
        out = self.linear2(out)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using hidden layer2
        out = self.linear3(out)

        return out


# Check --> weight and bias matrices
# output :
# torch.Size([32, 784])     = Weight matrix of layer1
# torch.Size([32])          = Bias matrix of layer1
# torch.Size([10, 32])      = Weight matrix of layer2
# torch.Size([10])          = Bias matrix of layer2
# for t in model.parameters():
#     print(t.shape)

# # Check --> Passing images to the custom nn layer
# for images, labels in train_dl:
#     outputs = model.forward(images)
#     loss = F.cross_entropy(outputs, labels)  # Cross entropy loss is calculated batch by batch
#     print('Loss :', loss)  # Loss is printed for the the batch
#     break  # Break iteration after one batch is processed


# GPU or CPU 
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor to a chosen device"""
    if isinstance(data, (list, tuple)):                     # Checking if the object is a list or tuple
        return [to_device(data, device) for x in data]      # Moving tensors to device. Either GPU or CPU
    return data.to(device, non_blocking=True)


# # Moving individual image tensors to the device. Either CPU or GPU
# for images, label in train_dl:
#     print(images.shape)
#     to_device(data=images, device=get_default_device())
#     print(images.device)
#     break


def accuracy(outputs, labels):
    # Getting the indexes of the outputs. Output with max probability is the number, therefore
    # obtaining the index of the number in the below step
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)


# Calculate loss for a batch of data
# optionally perform the gradient descent update step if an optimizer is provided.
# optionally computes a metric (e.g. accuracy) using the predictions and actual targets
# Parameters
# model = nn model
# xb = train input batch
# yb = corresponding labels of the train batch
def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    preds = model.forward(xb.reshape(batchsize, 784))
    loss = loss_func(preds, yb)

    if opt is not None:
        # Compute gradients
        loss.backward()
        # Update weights and biases
        opt.step()
        # Reset gradients to zero
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)

    # .item() converts tensor to floating point number
    return loss.item(), len(xb), metric_result


# Evaluation function
# Parameters
# model = model created
# loss_fn = the loss function used; "Cross entropy" or "log loss function" in this case
# valid_dl = validation data loader
# metric = metric used to compute goodness of fit. Accuracy in this case
def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        # Pass each batch through the created loss_batch function
        results = [loss_batch(model=model,
                              loss_func=loss_fn,
                              xb=xb,
                              yb=yb,
                              metric=metric)
                   for xb, yb in valid_dl]
        # separate the obtained losses, counts and metrics from loss_batch function
        losses, nums, metrics = zip(*results)
        # Total size of the dataset
        total = np.sum(nums)
        # Average loss across batches
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            # Avg of metric across batches
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric


def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric):
    for epoch in range(epochs):
        for xb, yb in train_dl:
            # Training
            loss, _, _ = loss_batch(model=model,
                                    loss_func=loss_fn,
                                    xb=xb,
                                    yb=yb,
                                    opt=opt,
                                    metric=metric)

        # Evaluation with the validation data loader
        result = evaluate(model=model, loss_fn=loss_fn, valid_dl=valid_dl, metric=accuracy)
        val_loss, total, val_acc = result

        print('Loss : {:.4f}, Accuracy : {:.4f}'.format(val_loss, val_acc))


print('--> Creating a model')
input_size = 784  # Input layer
num_class = 10  # output classes
model = MnistModel(in_size=input_size,
                   hidden_size1=32,
                   hidden_size2=16,
                   out_size=num_class)

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.02)
print('--> Calling fit method')

# Calling fit method
fit(epochs=2,
    model=model,
    loss_fn=F.cross_entropy,
    opt=optimizer,
    train_dl=train_dl,
    valid_dl=val_dl,
    metric=accuracy)



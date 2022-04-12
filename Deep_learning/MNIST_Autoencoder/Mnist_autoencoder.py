import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
from autoencoder_class import Autoencoder
import torch.nn as nn
import matplotlib.pyplot as plt

BATCH_SIZE = 2

# Downloading MNIST Dataset and saving it
mnist_data = MNIST(root='../../data/MNIST',
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]))

# Dataloader object
data_loader = DataLoader(dataset=mnist_data,
                        batch_size=BATCH_SIZE,
                        #shuffle=True
                        )

# Iterator
dataiter = iter(data_loader)
# Runs through the batches of data
data=dataiter.next() 


def accuracy(outputs, labels):
    """Getting the indexes of the outputs. Output with max probability is the number, therefore
    obtaining the index of the number in the below step
    Args:
        outputs: predicted values of putputs
        labels: Actual outputs
    Returns:
        accuracy of the predictions
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    """Calculate loss for a batch of data
    optionally perform the gradient descent update step if an optimizer is provided.
    optionally computes a metric (e.g. accuracy) using the predictions and actual targets
    Parameters
    Args:
        model: nn model
        xb: train input batch
        yb: corresponding labels of the train batch
        opt: optimizer
        metric: evaluation metric
    """
    preds = model.forward(xb.reshape(BATCH_SIZE, 784))
    # Note: loss_func can be of various types. (e.g. cross_entropy, mse, etc)
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

def fit(epochs, model, loss_fn, opt, train_dl, metric):
    """ Trains the model 
    Args:
        epochs: number of epochs for traing
        model = nn model
        xb: train input batch
        yb: corresponding labels of the train batc
        opt: optimizer
        metric: evaluation metric
    Returns: 

    """
    for epoch in range(epochs):
        for xb, yb in train_dl:
            # Training
            loss, _, _ = loss_batch(model=model,
                                    loss_func=loss_fn,
                                    xb=xb,
                                    yb=yb,
                                    opt=opt,
                                    metric=metric)


# # Verification
# data_flat=data[0][1].view(28,28) # Flattens and plots image tensor
# print(data_flat[1][1]) # prints number
# plt.savefig("test_img.png")

# Autoencoder object
# model = Autoencoder()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
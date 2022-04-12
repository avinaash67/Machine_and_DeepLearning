""" Functions needed for autoencoder"""
import torch

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def loss_batch(model, criterion, xb, yb, optimizer=None, metric=None):
    """Calculate loss for a batch of data
    optionally perform the gradient descent update step if an optimizer is provided.
    optionally computes a metric (e.g. accuracy) using the predictions and actual targets
    Parameters
    Args:
        model: nn model
        xb: batch of image tensor 
        yb: corresponding labels of the batch
        opt: optimizer
        metric: evaluation metric
    """
    # reshape mini-batch data to [N, 784] matrix
    # load it to the active device
    img_flat=xb.view(-1, 784)
    preds = model.forward(img_flat.to('cuda'))
    # Note: criterion can be of various types. (e.g. cross_entropy, mse, etc)
    # Loss is computed between the predicted image and the actual image
    # This is because, the autoencoder encodes the image to latent space and
    # The image is once again decoded(or redrawn) from the latent space
    loss = criterion(preds, img_flat.to('cuda'))  

    if optimizer is not None:
        # Compute gradients
        loss.backward()
        # Update weights and biases
        optimizer.step()
        # Reset gradients to zero
        optimizer.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)

    # .item() converts tensor to floating point number
    return loss.item(), len(xb), metric_result

def fit(epochs, model, criterion, optimizer, train_dl, test_dl, metric):
    """ Trains the model 
    Args:
        epochs: number of epochs for traing
        model: nn model
        xb: train input batch
        yb: corresponding labels of the train batc
        opt: optimizer
        metric: evaluation metric
    Returns: 

    """
    for epoch in range(epochs):
        for xb, yb in train_dl:
            # Moving to GPU
            xb.to('cuda')
            yb.to('cuda')
            # Training by batches
            loss, _, _ = loss_batch(model=model,
                                    criterion=criterion,
                                    xb=xb,
                                    yb=yb,
                                    optimizer=optimizer,
                                    metric=metric)
        print(f'Epoch:{epoch+1},Loss:{loss:.4f}')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow
import math


# Functions ----------------------------------------------------------------------------------------
# Sigmoid function
def sigmoid_numpy(x):
    return 1 / (1 + np.exp(-x))


# Loss function = log_loss_function
def log_loss_function(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, 1 - epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true * np.log(y_predicted_new) + (1 - y_true) * np.log(1 - y_predicted_new))


# Gradient descent implemented manually
def gradient_descent(age, affordability,y_true, epochs):
    # Assigning initial weights for inputs (Any arbitary value can be chosen)
    w1 = 1  # Weight for the age
    w2 = 1  # Weight for affordability
    bias = 0  # Bias

    for i in range(epochs):
        weighted_sum = w1 * age + w2 * affordability + bias  # gives the prediction y
        y_predicted = sigmoid_numpy(weighted_sum)

        loss = log_loss_function(y_true=y_true, y_predicted=y_predicted)


# ------------------------------------------------------------------------------------------------------

# Obtaining data
df = pd.read_csv("../data/insurance.csv")

# Splitting data into train and test dataset
X_train, X_test, Y_train, Y_split = train_test_split(df[['age', 'affordibility']], df[['bought_insurance']],
                                                     test_size=0.2)

# Scaling the input data between 0 and 1
X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age'] / 100
X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age'] / 100

gradient_descent(age=X_train_scaled['age'], affordability=X_train_scaled['affordibility'], epochs=1)

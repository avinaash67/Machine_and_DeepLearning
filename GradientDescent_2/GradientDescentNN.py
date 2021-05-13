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
def gradient_descent(age, affordability, y_true, epochs):
    # Assigning initial weights for inputs (Any arbitary value can be chosen)
    w1 = 1      # Weight for the age
    w2 = 1      # Weight for affordability
    bias = 0    # Bias
    rate = 0.5  # Learning rate

    for i in range(epochs):
        weighted_sum = w1 * age + w2 * affordability + bias  # Prediction y_hat
        y_predicted = sigmoid_numpy(weighted_sum)            # Prediction y_hat

        loss = log_loss_function(y_true=y_true, y_predicted=y_predicted)

        # Loss function used is the log loss
        # Calculate derivatives the variables or parameters in the log loss function
        # use partial derivatives to find how each individual parameter affects the loss function(log loss in this case)
        # The parameters present in the log loss function are W1 = Weight 1 ; W2 = Weight 2;

        w1_d = (1/len(age)) * np.dot(np.transpose(age), (y_predicted-y_true))
        w2_d = (1 / len(affordability)) * np.dot(np.transpose(affordability), (y_predicted - y_true))
        bias_d = np.mean(y_predicted-y_true)

        w1 = w1 - rate * w1_d
        w2 = w2 - rate * w2_d
        bias = bias - rate * bias_d

        print(w1,w2,bias,loss)

    return w1,w2,bias


# ------------------------------------------------------------------------------------------------------

# Obtaining data
df = pd.read_csv("../data/insurance.csv")

# Splitting data into train and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(df[['age', 'affordibility']], df[['bought_insurance']],
                                                     test_size=0.2)

# Scaling the input data between 0 and 1
X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age'] / 100
X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age'] / 100

w1, w2, bias = gradient_descent(age=X_train_scaled['age'],
                                affordability=X_train_scaled['affordibility'],
                                y_true=Y_train['bought_insurance'],
                                epochs=1000)

# print('weight1 = ', w1, 'weight2 = ', w2, 'bias = ', bias)

num=3
print(X_test['age'].iloc[num] *w1 + X_test['affordibility'].iloc[num] *w2 + bias)
print(sigmoid_numpy(X_test['age'].iloc[num] *w1 + X_test['affordibility'].iloc[num] *w2 + bias))

import numpy as np
import  tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------------------
# Load the handwritten numbers dataset; Splitting it into train and test dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# X_train is of shape 6000,28,28;
# 6000 represents the number of images (i.e. No of samples)
# Each image of size 28*28
# y_train contains array of integer numbers; This is used for training the neural network
print('Shape of y_train = ', y_train.shape)
# --------------------------------------------------------------------------------------------------------------
# Flattening the obtained data
X_train_flattened = X_train.reshape(len(X_train), 28*28)  # Converted to matrix of shape 60000,784
X_test_flattened = X_test.reshape(len(X_test), 28*28)    # Converted to matrix of shape 10000,784
print('Shape of X_train_flattened = ', X_train_flattened.shape)
# -------------------------------------------------------------------------------------------------------------
# Creating the neural network model
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
# Changing the parameters of the created model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

# Creating the final model
model.fit(X_train_flattened, y_train, epochs=2)
# -------------------------------------------------------------------------------------------------------------
# Evaluating the model
print('\n Evaluation = ', model.evaluate(X_test_flattened, y_test))

# -------------------------------------------------------------------------------------------------------------
# Printing the results
plt.matshow(X_test[2])

print(print('Shape of X_test_flattened = ', X_test_flattened.shape))
print('Shape of one element in X_test_flattened = ', X_test_flattened[2].shape)

# Predicting the output using the test data set (X_test_flattened)
output = model.predict(X_test_flattened)
print(output[2])

# printing the max value of the array
print('The number shown = ', np.argmax(output[2])) 

plt.show()








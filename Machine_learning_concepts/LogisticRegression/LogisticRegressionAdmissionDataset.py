import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy
import matplotlib.pyplot as plt

# Collecting data from csv
df = pd.read_csv("../../data/Admission_Predict.csv")

# X-Axis data (Independent variable)
X = df[["GRE Score", "TOEFL Score"]]
# Y-Axis data (Dependent variable)
y = df["Admission"]

# Data set is broken into two parts in a ratio of 75:25. It means 75% data will be used for model
# training and 25% for model testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Removing all non-zero numbers
y_train = numpy.nan_to_num(y_train)
X_train = numpy.nan_to_num(X_train)
X_test = numpy.nan_to_num(X_test)

# Instantiate the model (using the default parameters)
logistic_regression = LogisticRegression()

# Training the model
logistic_regression.fit(X_train, y_train)

# predicting the outcome if they will get admission
y_pred = logistic_regression.predict(X_test)

# plotting based on GRE score vs Admission
plt.scatter(X_test[:,0], y_pred)
# plotting based on IELTS score vs Admission
plt.scatter(X_test[:,1],y_pred)
plt.show()



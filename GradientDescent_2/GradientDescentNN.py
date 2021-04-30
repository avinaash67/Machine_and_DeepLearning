import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow

df = pd.read_csv("../data/insurance.csv")
# print(df.head())

X_train, X_test, Y_train, Y_split = train_test_split(df[['age', 'affordibility']],df.bought_insurance,test_size=0.2)


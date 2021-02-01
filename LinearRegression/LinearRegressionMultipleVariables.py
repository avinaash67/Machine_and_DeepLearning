import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Collecting data from csv
df = pd.read_csv("../data/homeprices.csv")
print(df)

# Creating a model
reg = linear_model.LinearRegression()  # Creating linear regression class object
reg.fit(df[['area', 'bedrooms', 'age']], df.price)

# predicted value yhat
print(reg.predict([[3000, 3, 40]]))

plt.xlabel('area(sqrft),bedrooms,age of the building')
plt.ylabel('price(US$)')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area', 'bedrooms', 'age']]), color='blue')
plt.show()

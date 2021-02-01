import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("../data/homeprices.csv")

plt.xlabel('area(sqrft)')
plt.ylabel('price(US$)')
plt.scatter(df.area, df.price, color='red', marker='+')


# Finding the values of m and b in y=m*x+b
# reg=model
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# Slope and intercept values
m = reg.coef_
b = reg.intercept_
x = 3000  # input values

# linear Regression Model
print(reg.predict([[3000]]))

# linear regression plot
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()




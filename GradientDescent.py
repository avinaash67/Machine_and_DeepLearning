import numpy as np


def gradient_descent(x, y):
    m_curr = 0
    b_curr = 0

    iterations = 1000

    n = len(x)
    learning_rate = 0.001

    for i in range(1000):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([ val**2 for val in (y-y_predicted)])

        md = -(2/n) * sum(x * (y-y_predicted))
        bd = -(2/n) * sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        print('m {}, b {}, cost or MSE {}, iteration{}'.format(m_curr, b_curr, cost, iterations))


x = np.array([1, 2, 3, 4, 5])  	# inputs
y = np.array([5, 7, 9, 11, 13]) 	# outputs

# calculate slope(m) and intercept(b) for the input data 
# Using the value of 'm' and 'b' one can calculate the estimated value of y

# y= mx+b Estimated value of y


gradient_descent(x, y)

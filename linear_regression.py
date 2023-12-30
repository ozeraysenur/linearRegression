import pandas as pd
import matplotlib.pyplot as plt

# Linear regression algorithm from scratch

# Example data for linear regression. It is aimed to observe and predict the effect of the year of experience,
# which is the independent variable, on the salary, which is the dependent variable.
data = pd.read_csv('Salary_dataset.csv')


def meansq_error(m, b, points):  # formula -> y = mx+b, points are the actual data points
    total_error = 0
    for i in range(len(points)):  # for loop to iterate over data points
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        total_error += (y - (m * x + b)) ** 2

    total_error / float(len(points))


def gradient_descent(m_first, b_first, points, learning_rate):  # Taking the partial derivative
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        m_gradient += -(2 / n) * x * (y - (m_first * x + b_first))
        b_gradient += -(2 / n) * (y - (m_first * x + b_first))
    m = m_first - m_gradient * learning_rate
    b = b_first - b_gradient * learning_rate

    return m, b


# Default values for m and b
m = 0
b = 0
learning_rate = 0.0001
epochs = 1000  # iterate over 1000 times
# Calling gradient descent method 1000 times to create our scatter plot
for i in range(epochs):
    m, b = gradient_descent(m, b, data, learning_rate)

print(m, b)
# Creating scatterplot using matplotlib library
plt.scatter(data.YearsExperience, data.Salary, color="purple")
plt.plot(list(range(0, 15)), [m * x + b for x in range(0, 15)], color="grey")
plt.show()

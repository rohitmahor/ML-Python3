# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values.reshape(-1, 1)


# splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X, y)


# fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)


# visualize data
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
fig = plt.figure('plymonial regression')
ax = fig.add_subplot(1, 1, 1)
ax1 = fig.add_subplot(1, 1, 1)
ax2 = fig.add_subplot(1, 1, 1)
ax.scatter(X, y, 10, 'r')
ax1.plot(X, lin_regressor.predict(X), 10, 'b')
ax2.plot(X_grid, poly_regressor.predict(poly_reg.fit_transform(X_grid)), 10, 'green')
plt.show()

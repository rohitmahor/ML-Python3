import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300)
regressor.fit(X, y)


# visualization
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
fig = plt.figure('Random Forest Regression')
ax = fig.add_subplot(1, 1, 1)
ax1 = fig.add_subplot(1, 1, 1)
ax.scatter(X, y, 10, 'r')
ax1.plot(X_grid, regressor.predict(X_grid), 10, 'b')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# fitting the decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# visualize model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
fig = plt.figure('decision tree regression')
ax = fig.add_subplot(1, 1, 1)
ax1 = fig.add_subplot(1, 1, 1)
ax.scatter(X, y, 10, 'r')
ax1.plot(X_grid, regressor.predict(X_grid), 10, 'b')
plt.show()
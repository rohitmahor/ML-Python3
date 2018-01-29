# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values


# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# fitting svr model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)


# visualize model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
fig = plt.figure('plymonial regression')
ax = fig.add_subplot(1, 1, 1)
ax1 = fig.add_subplot(1, 1, 1)
ax.scatter(X, y, 10, 'r')
ax1.plot(X_grid, regressor.predict(X_grid), 10, 'b')
plt.show()


# predict salary
y_pred = regressor.predict(sc_x.transform(6.5))
print(sc_y.inverse_transform(y_pred))

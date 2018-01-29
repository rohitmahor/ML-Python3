import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
y = y.reshape(-1, 1)
# print(X, y)

# create dummy variable
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelEncoder_x = LabelEncoder()
X[:, 3] = labelEncoder_x.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

# remove one dummy variable
X = X[:, 1:]


# splitting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# building optimal model using Backward elimination method
import statsmodels.formula.api as sm
X = np.append(np.ones((np.shape(X)[0], 1)).astype(int), X, axis=1)
X_test = np.append(np.ones((np.shape(X_test)[0], 1)).astype(int), X_test, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
print(regressor_OLS.summary())
y_pred1 = regressor_OLS.predict(X_test[:, [0, 1, 2, 3, 4, 5]])
# print(X)

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
print(regressor_OLS.summary())
y_pred2 = regressor_OLS.predict(X_test[:, [0, 1, 3, 4, 5]])

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
print(regressor_OLS.summary())
y_pred3 = regressor_OLS.predict(X_test[:, [0, 3, 4, 5]])

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
print(regressor_OLS.summary())
y_pred4 = regressor_OLS.predict(X_test[:, [0, 3, 5]])

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(y, X_opt).fit()
print(regressor_OLS.summary())
y_pred5 = regressor_OLS.predict(X_test[:, [0, 3]])
print(y_pred1)

# visualize data and model
fig = plt.figure('Multi linear regression model')
ax = fig.add_subplot(1, 1, 1)
ax1 = fig.add_subplot(1, 1, 1)
ax.scatter(X_train[:, 2], y_train, 10, 'r')
ax.scatter(X_test[:, 3], y_pred, 10, 'b')
ax.scatter(X_test[:, 3], y_pred1, 10, 'green')
ax.scatter(X_test[:, 3], y_pred2, 10, 'orange')
ax.scatter(X_test[:, 3], y_pred3, 10, 'cyan')
ax.scatter(X_test[:, 3], y_pred4, 10, 'pink')
ax.scatter(X_test[:, 3], y_pred5, 10, 'brown')
plt.show()
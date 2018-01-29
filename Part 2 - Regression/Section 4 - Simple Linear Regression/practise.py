# simple linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, -1].values

# splitting data into the trainging set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fitting simple linear regression
from sklearn.linear_model import LinearRegression
linearReg = LinearRegression()
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
linearReg.fit(X_train, y_train)

# predict the test set
y_pred = linearReg.predict(X_test)

# visualize model
fig6 = plt.figure('simple linear regression')
ax5 = fig6.add_subplot(1, 1, 1)
ax6 = fig6.add_subplot(1, 1, 1)
ax5.set_xlabel('year of experience')
ax5.set_ylabel('salary')
ax5.scatter(X, y, 10, 'r')
ax6.plot(X_train, linearReg.predict(X_train), 10, 'b')
plt.show()



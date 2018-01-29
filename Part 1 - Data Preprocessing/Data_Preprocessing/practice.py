import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, -1].values


# missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encoding Data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()


# Spliting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
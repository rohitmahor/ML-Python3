# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Mall_Customers.csv');
X = dataset.iloc[:, [3, 4]].values

# spitting datset
# feature scaling

# using elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmean = KMeans(n_clusters=i, random_state=0)
    kmean.fit(X)
    wcss.append(kmean.inertia_)

fig = plt.figure('Elbow method')
ax = fig.add_subplot(1, 1, 1)
plt.plot(range(1, 11), wcss)
plt.legend()
plt.show()


# fitting model
kmean = KMeans(n_clusters=5, random_state=0)
# kmean.fit(X)
y_kmean = kmean.fit_predict(X)

# visualize model
plt.scatter(X[y_kmean == 0, 0], X[y_kmean == 0, 1], s=100, c='red', label='cluster 1')
plt.scatter(X[y_kmean == 1, 0], X[y_kmean == 1, 1], s=100, c='pink', label='cluster 2')
plt.scatter(X[y_kmean == 2, 0], X[y_kmean == 2, 1], s=100, c='yellow', label='cluster 3')
plt.scatter(X[y_kmean == 3, 0], X[y_kmean == 3, 1], s=100, c='green', label='cluster 4')
plt.scatter(X[y_kmean == 4, 0], X[y_kmean == 4, 1], s=100, c='cyan', label='cluster 5')
plt.legend()
plt.show()

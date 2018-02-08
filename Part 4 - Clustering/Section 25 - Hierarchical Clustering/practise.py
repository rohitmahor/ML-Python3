# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# plot dendogram
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('customers')
plt.ylabel('Euclidean distances')
plt.show()


# fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)


# visualize y_hc
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], c='red', label='cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], c='yellow', label='cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], c='blue', label='cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], c='pink', label='cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], c='green', label='cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
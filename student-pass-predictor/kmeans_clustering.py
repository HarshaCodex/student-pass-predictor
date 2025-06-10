import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
    [8, 2],
    [10, 2],
    [9, 3],
])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)

print(kmeans.labels_)

colors = ["red", "blue"]

for i in range(len(data)):
    plt.scatter(data[i][0], data[i][1], color=colors[kmeans.labels_[i]])

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], color="green", marker="x", s=200)

plt.title("K-Means Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
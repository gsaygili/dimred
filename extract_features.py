import matplotlib.pyplot as plt
import numpy as np
import calc_error as err
import scipy.spatial.distance as dist
import time

emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
Xe = np.load(emb_folder+"Xemb_0.npy")
y = np.load(y_folder+"y_5000_0.npy")
X = np.load(y_folder+"X_5000_0.npy")
X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))

K = 20
features = np.zeros((X.shape[0], K, 5))
X_d = dist.squareform(dist.pdist(Xe, "euclidean"))
sort_index_d = np.argsort(X_d)

# calculate euclidean distance
print("Calculating Euclidean Distances")
start_time = time.time()
X_D = dist.squareform(dist.pdist(X, "euclidean"))
sort_D = np.sort(X_D)
for i in range(X_D.shape[0]):
    cost = X_D[i, :]
    s_D = sort_D[i, 1:K+1]
    s_d = cost[sort_index_d[i, 1:K+1]]
    features[i, :, 0] = np.abs(s_D-s_d)
print("--- Euclidean takes: %s seconds ---" % (time.time() - start_time))

# calculate cosine distance
print("Calculating Cosine Distances")
start_time = time.time()
X_D = dist.squareform(dist.pdist(X, "cosine"))
sort_D = np.sort(X_D)
for i in range(X_D.shape[0]):
    cost = X_D[i, :]
    s_D = sort_D[i, 1:K+1]
    s_d = cost[sort_index_d[i, 1:K+1]]
    features[i, :, 1] = np.abs(s_D-s_d)
print("--- Cosine takes: %s seconds ---" % (time.time() - start_time))

# calculate minkowski distance 1 (mahalonabis)
print("Calculating standardized Euclidean distance Distances")
start_time = time.time()
X_D = dist.squareform(dist.pdist(X, 'seuclidean', V=None))
# X_D = dist.squareform(dist.pdist(X, "minkowski", p=1))
sort_D = np.sort(X_D)
for i in range(X_D.shape[0]):
    cost = X_D[i, :]
    s_D = sort_D[i, 1:K+1]
    s_d = cost[sort_index_d[i, 1:K+1]]
    features[i, :, 2] = np.abs(s_D-s_d)
print("--- standardized Euclidean distance takes: %s seconds ---" % (time.time() - start_time))

# calculate minkowski distance 3
print("Calculating correlation Distances")
start_time = time.time()
# X_D = dist.squareform(dist.pdist(X, "minkowski", p=3))
X_D = dist.squareform(dist.pdist(X, 'correlation'))
sort_D = np.sort(X_D)
for i in range(X_D.shape[0]):
    cost = X_D[i, :]
    s_D = sort_D[i, 1:K+1]
    s_d = cost[sort_index_d[i, 1:K+1]]
    features[i, :, 3] = np.abs(s_D-s_d)
print("--- correlation takes: %s seconds ---" % (time.time() - start_time))

# calculate chebyshev distance
print("Calculating Chebyshev Distances")
start_time = time.time()
X_D = dist.squareform(dist.pdist(X, "chebyshev"))
sort_D = np.sort(X_D)
for i in range(X_D.shape[0]):
    cost = X_D[i, :]
    s_D = sort_D[i, 1:K+1]
    s_d = cost[sort_index_d[i, 1:K+1]]
    features[i, :, 4] = np.abs(s_D-s_d)
print("--- Chebyshev takes: %s seconds ---" % (time.time() - start_time))

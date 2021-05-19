import matplotlib.pyplot as plt
import numpy as np
import calc_error as err
import scipy.spatial.distance as dist


emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
Xe = np.load(emb_folder+"Xemb_0.npy")
y = np.load(y_folder+"y_5000_0.npy")
X = np.load(y_folder+"X_5000_0.npy")
X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))

# calculate euclidean distances in D
K = 20
features = np.zeros((X.shape[0], K, 5))
X_D = dist.squareform(dist.pdist(X, "euclidean"))
X_d = dist.squareform(dist.pdist(Xe, "euclidean"))
sort_D = np.sort(X_D)
sort_index_d = np.argsort(X_d)

for i in range(X_D.shape[0]):
    cost = X_D[i, :]
    s_D = sort_D[i, 1:K+1]
    s_d = cost[sort_index_d[i, 1:K+1]]
    features[i, :, 0] = np.abs(s_D-s_d)

# calculate cosine distance
X_D = dist.squareform(dist.pdist(X, "cosine"))
sort_D = np.sort(X_D)
for i in range(X_D.shape[0]):
    cost = X_D[i, :]
    s_D = sort_D[i, 1:K+1]
    s_d = cost[sort_index_d[i, 1:K+1]]
    features[i, :, 1] = np.abs(s_D-s_d)

# calculate minkowski distance 1 (mahalonabis)

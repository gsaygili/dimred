# Block aggregation of cost spaces in a neighborhood

import numpy as np
from sys import platform
import scipy.spatial.distance as dist


def calc_block_cost(B=10, sample_id=0):
    if platform == "linux" or platform == "linux2":
        emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
        y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
    elif platform == "darwin":
        emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
        y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
    elif platform == "win32":
        emb_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/emb_p30/"
        y_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/"

    X_embd = np.load(emb_folder + "Xemb_" + str(sample_id) + ".npy")
    x_cost = np.load(emb_folder + "features_" + str(sample_id) + ".npy")

    # block size
    avg_features = np.zeros(x_cost.shape)
    K = x_cost.shape[1]
    X_d = dist.squareform(dist.pdist(X_embd, "euclidean"))
    sort_index_d = np.argsort(X_d)[:, 1:]
    for k in range(x_cost.shape[2]):
        for i in range(x_cost.shape[0]):  # for all sample
            neighs = sort_index_d[i, 0:B]
            avg_features[i, :, k] = np.mean(x_cost[neighs, :, k], axis=0)

    np.save(emb_folder + "avg_features_" + str(sample_id) + '_blocksize_' + str(B), avg_features)


for i in range(12):
    print("iteration id: ", str(i))
    calc_block_cost(B=20, sample_id=i)


import matplotlib.pyplot as plt
import numpy as np
import calc_error as err
import scipy.spatial.distance as dist
import time
from sklearn import preprocessing
from sys import platform


def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = np.max(arr) - np.min(arr)
    for i in arr:
        temp = (((i - np.min(arr))*diff)/diff_arr) + t_min
        if np.isnan(temp):
            temp = t_max
        norm_arr.append(temp)
    return np.array(norm_arr)


def extract_euc_seuc_corr_cheby_feats(K=20, sample_id=0):
    mm_scaler = preprocessing.MinMaxScaler()
    if platform == "linux" or platform == "linux2":
        emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
        y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
    elif platform == "darwin":
        emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
        y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
    elif platform == "win32":
        emb_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/emb_p30/"
        y_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/"

    Xe = np.load(emb_folder + "Xemb_" + str(sample_id) + ".npy")
    y = np.load(y_folder + "y_5000_" + str(sample_id) + ".npy")
    X = np.load(y_folder + "X_5000_" + str(sample_id) + ".npy")
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

    features = np.zeros((X.shape[0], K, 7))
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
        features[i, :, 0] = normalize(np.abs(s_D - s_d))

    print(np.any(np.isnan(features)))
    print("--- Euclidean takes: %s seconds ---" % (time.time() - start_time))

    # calculate euclidean distance
    print("Calculating Standardized Euclidean Distances")
    start_time = time.time()
    X_D = dist.squareform(dist.pdist(X, "seuclidean"))
    sort_D = np.sort(X_D)
    for i in range(X_D.shape[0]):
        cost = X_D[i, :]
        s_D = sort_D[i, 1:K + 1]
        s_d = cost[sort_index_d[i, 1:K + 1]]
        features[i, :, 1] = normalize(np.abs(s_D - s_d))

    print(np.any(np.isnan(features)))
    print("--- Standardized Euclidean takes: %s seconds ---" % (time.time() - start_time))

    # calculate cosine distance
    print("Calculating Cosine Distances")
    start_time = time.time()
    X_D = dist.squareform(dist.pdist(X, "cosine"))
    sort_D = np.sort(X_D)
    for i in range(X_D.shape[0]):
        cost = X_D[i, :]
        s_D = sort_D[i, 1:K+1]
        s_d = cost[sort_index_d[i, 1:K+1]]
        features[i, :, 2] = normalize(np.abs(s_D - s_d))

    # normalize the scores
    print(np.any(np.isnan(features)))
    print("--- Cosine takes: %s seconds ---" % (time.time() - start_time))

    # calculate correlation
    start_time = time.time()
    X_D = dist.squareform(dist.pdist(X, "correlation"))
    sort_D = np.sort(X_D)
    for i in range(X_D.shape[0]):
        cost = X_D[i, :]
        s_D = sort_D[i, 1:K + 1]
        s_d = cost[sort_index_d[i, 1:K + 1]]
        features[i, :, 3] = normalize(np.abs(s_D - s_d))

    print(np.any(np.isnan(features)))
    print("--- correlation takes: %s seconds ---" % (time.time() - start_time))

    # calculate chebyshev
    start_time = time.time()
    X_D = dist.squareform(dist.pdist(X, "chebyshev"))
    sort_D = np.sort(X_D)
    for i in range(X_D.shape[0]):
        cost = X_D[i, :]
        s_D = sort_D[i, 1:K + 1]
        s_d = cost[sort_index_d[i, 1:K + 1]]
        features[i, :, 4] = normalize(np.abs(s_D - s_d))

    print(np.any(np.isnan(features)))
    print("--- chebyshev takes: %s seconds ---" % (time.time() - start_time))

    # calculate canberra
    start_time = time.time()
    X_D = dist.squareform(dist.pdist(X, "canberra"))
    sort_D = np.sort(X_D)
    for i in range(X_D.shape[0]):
        cost = X_D[i, :]
        s_D = sort_D[i, 1:K + 1]
        s_d = cost[sort_index_d[i, 1:K + 1]]
        features[i, :, 5] = normalize(np.abs(s_D - s_d))
    print(np.any(np.isnan(features)))
    print("--- canberra takes: %s seconds ---" % (time.time() - start_time))

    # calculate braycurtis
    start_time = time.time()
    X_D = dist.squareform(dist.pdist(X, "braycurtis"))
    sort_D = np.sort(X_D)
    for i in range(X_D.shape[0]):
        cost = X_D[i, :]
        s_D = sort_D[i, 1:K + 1]
        s_d = cost[sort_index_d[i, 1:K + 1]]
        features[i, :, 6] = normalize(np.abs(s_D - s_d))
    print(np.any(np.isnan(features)))
    print("--- braycurtis takes: %s seconds ---" % (time.time() - start_time))

    np.save(emb_folder + "features_" + str(sample_id), features)


for i in range(12):
    print("iteration id: ", str(i))
    extract_euc_seuc_corr_cheby_feats(K=20, sample_id=i)


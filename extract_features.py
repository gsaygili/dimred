import numpy as np
import scipy.spatial.distance as dist
import time
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


def normalize_mat(mat, t_min=0, t_max=1):
    arr = mat.flatten()
    norm_arr = normalize(arr, t_min, t_max)
    return norm_arr.reshape(mat.shape)


def extract_feats(K=20, sample_id=0,
                  distance_measures=["euclidean", "cosine", "correlation", "chebyshev", "canberra", "braycurtis"]):
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
    X = np.load(y_folder + "X_5000_" + str(sample_id) + ".npy")
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

    features = np.zeros((X.shape[0], K, len(distance_measures)))
    X_d = dist.squareform(dist.pdist(Xe, "euclidean"))
    sort_index_d = np.argsort(X_d)

    for ind, c in enumerate(distance_measures):
        print("Calculating " + c + " Distances")
        start_time = time.time()
        X_D = dist.squareform(dist.pdist(X, c))
        X_D = np.nan_to_num(X_D)
        X_D = (X_D - np.min(X_D))/np.ptp(X_D)
        sort_D = np.sort(X_D)
        for i in range(X_D.shape[0]):
            cost = X_D[i, :]
            s_D = sort_D[i, 1:K + 1]
            s_d = cost[sort_index_d[i, 1:K + 1]]
            features[i, :, ind] = np.abs(s_D - s_d)

        print(np.any(np.isnan(features)))
        print("--- " + c + " takes: %s seconds ---" % (time.time() - start_time))

    np.save(emb_folder + "features_" + str(sample_id), features)


for i in range(12):
    print("iteration id: ", str(i))
    extract_feats(K=20, sample_id=i)


import numpy as np
import scipy.spatial.distance as dist


def find_errors_majority(X_emb, labels, K=20):
    # take a closest neighborhood of K and check whether the label of the majority is the same as the center
    X_d = dist.squareform(dist.pdist(X_emb, "euclidean"))
    # the first column is the sample itself since the distance is zero
    sort_index = np.argsort(X_d)
    error_list = []
    for i in range(X_d.shape[0]):
        K_neigh = sort_index[i, 1:K+1]
        lab_neigh = labels[K_neigh]
        num_label = np.count_nonzero(lab_neigh == labels[i])
        if num_label < K/2:
            error_list.append(i)
    err = np.array(error_list)
    return err


def find_errors_nearest(X_emb, labels):
    # take a closest neighborhood of K and check whether the label of the majority is the same as the center
    X_d = dist.squareform(dist.pdist(X_emb, "euclidean"))
    # the first column is the sample itself since the distance is zero
    sort_index = np.argsort(X_d)
    error_list = []
    for i in range(X_d.shape[0]):
        nearest_neigh = labels[sort_index[i, 1]]
        if labels[i] != nearest_neigh:
            error_list.append(i)
    err = np.array(error_list)
    return err


def find_worst_best_N(X_emb, labels, K=20, N=10):
    # take a closest neighborhood of K and check whether the label of the majority is the same as the center
    X_d = dist.squareform(dist.pdist(X_emb, "euclidean"))
    corr_neigh_num = np.zeros(X_emb.shape[0])
    # the first column is the sample itself since the distance is zero
    sort_index = np.argsort(X_d)
    error_list = []
    for i in range(X_d.shape[0]):
        K_neigh = sort_index[i, 1:K+1]
        lab_neigh = labels[K_neigh]
        num_label = np.count_nonzero(lab_neigh == labels[i])  # dogru komsu sayisi
        corr_neigh_num[i] = num_label
    bestN = np.argsort(corr_neigh_num)[-N:]
    worstN = np.argsort(corr_neigh_num)[:N]
    return worstN, bestN


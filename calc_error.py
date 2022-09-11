import numpy as np
import scipy.spatial.distance as dist

# functions for calculating error
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

def find_error_score(X_emb,labels,K=20):
    # take a closest neighborhood of K and check whether the label of the majority is the same as the center
    X_d = dist.squareform(dist.pdist(X_emb, "euclidean"))
    # the first column is the sample itself since the distance is zero
    sort_index = np.argsort(X_d)
    error_list = []
    Score = []
    for i in range(X_d.shape[0]):
        K_neigh = sort_index[i, 1:K+1]
        lab_neigh = labels[K_neigh]
        num_label = np.count_nonzero(lab_neigh == labels[i])
        if num_label < K/2:
            error_list.append(i)
        score=num_label/K
        Score.append(score)
    err = np.array(error_list)
    return err,Score
    
def calc_npr(X, X_emb, K=20):
    X_d = dist.squareform(dist.pdist(X_emb, "euclidean"))
    X_D = dist.squareform(dist.pdist(X, "euclidean"))
    ind_d = np.argsort(X_d)
    ind_D = np.argsort(X_D)
    npr = np.zeros(X_d.shape[0],)
    for i in range(X_d.shape[0]):
        K_d = ind_d[i, 1:K+1]
        K_D = ind_D[i, 1:K+1]
        inter = np.intersect1d(K_d, K_D)
        count = inter.shape[0]
        npr[i] = count/K
    return npr
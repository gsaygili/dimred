import numpy as np
import apply_tsne as at
import scipy.spatial.distance as dist

emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
Xe = np.load(emb_folder+"Xemb_0.npy")
y = np.load(y_folder+"y_5000_0.npy")
# at.plot_embedding(Xe, y)


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
        nearest_neigh = y[sort_index[i, 1]]
        if y[i] != nearest_neigh:
            error_list.append(i)
    err = np.array(error_list)
    return err


# plot errors
errs = find_errors_nearest(Xe, y)
at.plot_embedding_with_errors(Xe, y, errs)

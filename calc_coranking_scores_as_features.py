import coranking_scores as crank
from sys import platform
import numpy as np
import time
import scipy.spatial.distance as dist
from coranking import coranking_matrix
from coranking.metrics import trustworthiness, continuity, LCMC


def coranking_matrix_(high_data, low_data):
    """Generate a co-ranking matrix from two data frames of high and low
    dimensional data.

    :param high_data: DataFrame containing the higher dimensional data.
    :param low_data: DataFrame containing the lower dimensional data.
    :returns: the co-ranking matrix of the two data sets.
    """
    n, m = high_data.shape
    high_distance = dist.squareform(dist.pdist(high_data))
    low_distance = dist.squareform(dist.pdist(low_data))

    high_ranking = high_distance.argsort(axis=1).argsort(axis=1)
    low_ranking = low_distance.argsort(axis=1).argsort(axis=1)

    Q, xedges, yedges = np.histogram2d(high_ranking.flatten(),
                                       low_ranking.flatten(),
                                       bins=n)

    #Q = Q[1:, 1:]  # remove rankings which correspond to themselves
    return Q


if platform == "linux" or platform == "linux2":
    emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
elif platform == "darwin":
    emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
elif platform == "win32":
    emb_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/"


sample_id = 0
Xe = np.load(emb_folder + "Xemb_" + str(sample_id) + ".npy")
X = np.load(y_folder + "X_5000_" + str(sample_id) + ".npy")
X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

print('calculating Euclidean distances')
start_time = time.time()
X_d = dist.squareform(dist.pdist(Xe, "euclidean"))
X_D = dist.squareform(dist.pdist(X, "euclidean"))
print("--- Euclidean distance takes: %s seconds ---" % (time.time() - start_time))


print("Calculating the Q - coranking matrix")
Q = coranking_matrix_(X, Xe)
print("--- co-ranking takes: %s seconds ---" % (time.time() - start_time))

print("Extracting coranking features")
T, C, QNN, AUC, LCMC, kmax, Qlocal, Qglobal = crank.coranking_matrix_metrics(Q)
print("--- co-ranking features takes: %s seconds ---" % (time.time() - start_time))

# print("Calculating ranking matrices")
# start_time = time.time()
# rank_D = crank.ranking_matrix(X_D)
# print('rank 1 finished')
# rank_d = crank.ranking_matrix(X_d)
# print("--- ranking takes: %s seconds ---" % (time.time() - start_time))
#
# print("Calculating the Q - coranking matrix")
# start_time = time.time()
# Q = crank.coranking_matrix(rank_D, rank_d)
# print("--- co-ranking takes: %s seconds ---" % (time.time() - start_time))
#
# print("Extracting coranking features")
# start_time = time.time()
# T, C, QNN, AUC, LCMC, kmax, Qlocal, Qglobal = crank.coranking_matrix_metrics(Q)
# print("--- co-ranking features takes: %s seconds ---" % (time.time() - start_time))
#
# # save path
# np.save(emb_folder + "corank_T_" + str(sample_id), T)
# np.save(emb_folder + "corank_C_" + str(sample_id), C)
# np.save(emb_folder + "corank_QNN_" + str(sample_id), QNN)
# np.save(emb_folder + "corank_AUC_" + str(sample_id), AUC)
# np.save(emb_folder + "corank_LCMC_" + str(sample_id), LCMC)
# np.save(emb_folder + "corank_kmax_" + str(sample_id), kmax)
# np.save(emb_folder + "corank_Qlocal_" + str(sample_id), Qlocal)
# np.save(emb_folder + "corank_Qglobal_" + str(sample_id), Qglobal)

import coranking_scores as crank
from sys import platform
import numpy as np
import time
import scipy.spatial.distance as dist


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

print("Calculating ranking matrices")
start_time = time.time()
rank_D = crank.ranking_matrix(X_D)
print('rank 1 finished')
rank_d = crank.ranking_matrix(X_d)
print("--- ranking takes: %s seconds ---" % (time.time() - start_time))

print("Calculating the Q - coranking matrix")
start_time = time.time()
Q = crank.coranking_matrix(rank_D, rank_d)
print("--- co-ranking takes: %s seconds ---" % (time.time() - start_time))

print("Extracting coranking features")
start_time = time.time()
T, C, QNN, AUC, LCMC, kmax, Qlocal, Qglobal = crank.coranking_matrix_metrics(Q)
print("--- co-ranking features takes: %s seconds ---" % (time.time() - start_time))

# save path
np.save(emb_folder + "corank_T_" + str(sample_id), T)
np.save(emb_folder + "corank_C_" + str(sample_id), C)
np.save(emb_folder + "corank_QNN_" + str(sample_id), QNN)
np.save(emb_folder + "corank_AUC_" + str(sample_id), AUC)
np.save(emb_folder + "corank_LCMC_" + str(sample_id), LCMC)
np.save(emb_folder + "corank_kmax_" + str(sample_id), kmax)
np.save(emb_folder + "corank_Qlocal_" + str(sample_id), Qlocal)
np.save(emb_folder + "corank_Qglobal_" + str(sample_id), Qglobal)

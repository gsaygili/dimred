import matplotlib.pyplot as plt
import numpy as np
from sys import platform
import calc_error as cerr
import plot_embedding as plt_emb


if platform == "linux" or platform == "linux2":
    emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
elif platform == "darwin":
    emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
elif platform == "win32":
    emb_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/"


# plot the n best embbedded samples and n worst embedded samples
X_embd = np.load(emb_folder + "Xemb_0.npy")
x_cost = np.load(emb_folder + "features_euc_corr_0.npy")
y_labl = np.load(y_folder + "y_5000_0.npy")

worstN, bestN = cerr.find_worst_best_N(X_embd, y_labl, K=20, N=10)
plt_emb.plot_embedding_with_errors_and_corrects(X_embd, y_labl, worstN, bestN)

# plot cost spaces


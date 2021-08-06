import matplotlib.pyplot as plt
import numpy as np
from sys import platform


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
X_tr = np.load(emb_folder + "Xemb_0.npy")
x_train = np.load(emb_folder + "features_euc_corr_0.npy")
y_tr = np.load(y_folder + "y_5000_0.npy")



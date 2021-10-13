# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:12:09 2021

@author: Admin
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers import LSTM, ConvLSTM2D, MaxPooling2D
import keras
from keras import optimizers
import scipy.io as sio
from sys import platform
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import scipy.spatial.distance as dist
import time
from sys import platform
import calc_error as err
#extract features from test set

def extract_feats(K=20, sample_id=0,
                  distance_measures=["euclidean", "cosine", "correlation", "chebyshev", "canberra", "braycurtis"]):
    if platform == "linux" or platform == "linux2":
        emb_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/Ovarian-PBSII-061902/emb_p30/"
        y_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/Ovarian-PBSII-061902/emb_p30/"
    elif platform == "darwin":
        emb_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/emb_p30/"
        y_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/emb_p30/"
    elif platform == "win32":
        emb_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/Ovarian-PBSII-061902/emb_p30/"
        y_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/Ovarian-PBSII-061902/emb_p30/"

    Xe = np.load(emb_folder + "Xemb.npy")
    X = np.load(y_folder + "X.npy")
#    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

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

    np.save(emb_folder + "features_0.npy", features)
extract_feats()
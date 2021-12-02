#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:25:15 2021

@author: busraozgode
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

#extract features from test set

def extract_feats(K=20, sample_id=0,
                  distance_measures=["euclidean", "cosine", "correlation", "chebyshev", "canberra", "braycurtis"]):

    Xe = np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/AMB_integrated/X_emb.npy")
#    X = np.load(y_folder + "X_5000_" + str(sample_id) + ".npy")
    X = np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/AMB_integrated/X.npy")
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
            s_d = np.sort(cost[sort_index_d[i, 1:K + 1]])
            features[i, :, ind] = np.abs(s_D - s_d)

        print(np.any(np.isnan(features)))
        print("--- " + c + " takes: %s seconds ---" % (time.time() - start_time))

    np.save("/Users/busraozgode/Desktop/t-SNE/Datasets/AMB_integrated/features_0.npy", features)
    
#for i in range(120):
#    print("iteration id: ", str(i))
#    extract_feats(K=20, sample_id=i)

extract_feats()

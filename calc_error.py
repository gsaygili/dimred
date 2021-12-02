#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:38:08 2021

@author: busraozgode
"""

import numpy as np
from keras import optimizers
from sys import platform
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import scipy.spatial.distance as dist
import time
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from scipy.stats import variation
import keras
import scipy.io as sio
from scipy.optimize import curve_fit
##find_err_functions
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
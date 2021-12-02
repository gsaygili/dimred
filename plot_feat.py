#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 17:04:11 2021

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
from matplotlib import pyplot as plt
import numpy as np


def plot_embedding(X_d, y):
    labels = str(y).strip('[]')
    plt.figure()
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'olive', 'orange', 'purple'
    print(type(colors))
    for i in np.unique(y):
        plt.scatter(X_d[np.where(y == i), 0], X_d[np.where(y == i), 1], c=colors[i], s=3)
    # for i, c in zip(y, colors):
    #     plt.scatter(X_d[y == i, 0], X_d[y == i, 1], c=c, s=3)
    # plt.scatter(X_d[15, 0], X_d[15, 1], color="none", edgecolor="red")
    plt.show()


def plot_embedding_with_errors(X_d, y, err_list):
    labels = str(y).strip('[]')
    plt.figure()
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'olive', 'orange', 'purple'
    for i in np.unique(y):
        plt.scatter(X_d[np.where(y == i), 0], X_d[np.where(y == i), 1], c=colors[i], s=3)
    # for i, c in zip(y, colors):
    #     plt.scatter(X_d[y == i, 0], X_d[y == i, 1], c=c, s=3)
    for i in range(err_list.shape[0]):
        plt.scatter(X_d[err_list[i], 0], X_d[err_list[i], 1], label='Example legend entry.', s=80, marker=r'o',
                    facecolors='none',
                    edgecolors='red')
    plt.show()


def plot_embedding_with_errors_and_corrects(X_d, y, err_list, corr_list):
    labels = str(y).strip('[]')   
    plt.figure()
    ax2 = plt.gca()
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'olive', 'k', 'orange', 'purple'
    for i in np.unique(y):
        plt.scatter(X_d[np.where(y == i), 0], X_d[np.where(y == i), 1], c=colors[i], s=3)
    # for i, c in zip(y, colors):
    #     plt.scatter(X_d[y == i, 0], X_d[y == i, 1], c=c, s=3)
    for i in range(corr_list.shape[0]):
        plt.scatter(X_d[corr_list[i], 0], X_d[corr_list[i], 1], label='Example legend entry.', s=80, marker=r'o',
                    facecolors='none',
                    edgecolors='red', linewidths=2)
    ax2.axis("off")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)   
    plt.savefig("/Users/busraozgode/Desktop/t-SNE/emb_with_correct_2.pdf", bbox_inches="tight")
    plt.show()
    
def plot_embedding_with_worstN_bestN(X_d, y, worst_list, best_list):
    plt.figure()
    ax = plt.gca()
    colors = 'olive', 'g', 'b', 'c', 'm', 'y','r' , 'k', 'orange', 'purple'
    for i in np.unique(y):
        plt.scatter(X_d[np.where(y == i), 0], X_d[np.where(y == i), 1], c=colors[i], s=3)
    # for i, c in zip(y, colors):
    #     plt.scatter(X_d[y == i, 0], X_d[y == i, 1], c=c, s=3)
    for i in range(worst_list.shape[0]):
        plt.scatter(X_d[worst_list[i], 0], X_d[worst_list[i], 1], label='Example legend entry.', s=80, marker=r'o',
                    facecolors='none',
                    edgecolors='red', linewidths=2)  
    for i in range(best_list.shape[0]):
        plt.scatter(X_d[best_list[i], 0], X_d[best_list[i], 1], label='Example legend entry.', s=80, marker=r'o',
                    facecolors='none',
                    edgecolors='green', linewidths=2)    
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axis("off")
    plt.savefig("/Users/busraozgode/Desktop/t-SNE/emb_with_worst_best.pdf", bbox_inches="tight")
    plt.show()
    
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


def find_worst_best_N(X, X_emb, labels, K=20, N=10):
    # take a closest neighborhood of K and check whether the label of the majority is the same as the center
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
    bestN = np.argsort(npr)[-N:]
    worstN = np.argsort(npr)[:N]
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

from sys import platform

if platform == "linux" or platform == "linux2":
    emb_folder = "/Users/busraozgode/Desktop/t-SNE/Datasets/mnist_test/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
elif platform == "darwin":
    emb_folder = "/Users/busraozgode/Desktop/t-SNE/Datasets/mnist_test/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
elif platform == "win32":
    emb_folder = "/Users/busraozgode/Desktop/t-SNE/Datasets/mnist_test/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"


# plot the n best embbedded samples and n worst embedded samples

X_embd = np.load(emb_folder + "X_emb_0.npy")
x_cost = np.load(emb_folder + "features_0.npy")
y_labl = np.load(emb_folder + "y_test.npy")

        
K = 20
X = np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/mnist_test/X_test.npy")
X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

X_d = dist.squareform(dist.pdist(X_embd, "euclidean"))
sort_index_d = np.argsort(X_d)

X_D = dist.squareform(dist.pdist(X, "euclidean"))
X_D = np.nan_to_num(X_D)
X_D = (X_D - np.min(X_D))/np.ptp(X_D)
sort_D = np.sort(X_D)

N = 20
worstN, bestN = find_worst_best_N(X, X_embd, y_labl, K=20, N=N)
plot_embedding_with_worstN_bestN(X_embd, y_labl, np.ravel(worstN[6,]), np.ravel(bestN[0,]))

# Plot Euclidean Distance Differences Graph between low-dimensional and high-dimensional space closest neighbors

id2 = 6
cost = X_D[worstN[id2], :]
s_D = sort_D[worstN[id2], 1:K + 1]
s_d = np.sort(cost[sort_index_d[worstN[id2], 1:K + 1]])

id = 0
cost2 = X_D[bestN[id], :]
s_D2 = sort_D[bestN[id], 1:K + 1]
s_d2 = np.sort(cost2[sort_index_d[bestN[id], 1:K + 1]])
from matplotlib import ticker
fig= plt.figure(figsize =(10, 7))
plt.rc('font', size=20)
ax=plt.axes()
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
plt.plot(s_D, label=r'$D(x_i,N_D(x_i,k))$')
plt.plot(s_d, label=r'$D(x_i,N_d(x_i,k))$')
plt.legend(loc='upper left',prop={"size":27})
plt.xticks([0, 10, 20])
plt.ylabel('Normalized Euclidean Distance')
plt.xlabel('k')
plt.savefig("/Users/busraozgode/Desktop/t-SNE/euclidean_graph_error_2.pdf",
            bbox_inches="tight")
plt.show()


fig = plt.figure(figsize =(10, 7))
plt.rc('font', size=20)
xi = list(range(len(s_D)))
plt.plot(s_D2, label=r'$D(x_i,N_D(x_i,k))$')
plt.plot(s_d2, label=r'$D(x_i,N_d(x_i,k))$')
plt.legend(loc='upper left', prop={"size":27})
plt.xticks([0,10,20])
plt.ylabel('Normalized Euclidean Distance')
plt.xlabel('k')
plt.savefig("/Users/busraozgode/Desktop/t-SNE/euclidean_graph_correct_2.pdf", bbox_inches="tight")
plt.show()

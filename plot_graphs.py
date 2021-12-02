#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:55:14 2021

@author: busraozgode
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import glob
from pathlib import Path
import scipy.spatial.distance as dist

def tsne_p(data, dim=2, perplexity=30):
    X = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
    X_embedded = TSNE(n_components=dim, perplexity=perplexity).fit_transform(X)
    return X_embedded

def plot_embedding_with_errors(X_d, y, err_list):
    labels = str(y).strip('[]')
    plt.figure()
    ax = plt.gca()
    colors =  'purple', 'g', 'b', 'orange', 'c', 'm', 'y', 'k', 'lightpink','olive'
    for i in np.unique(y):
        plt.scatter(X_d[np.where(y == i), 0], X_d[np.where(y == i), 1], c=colors[i], s=5)
    # for i, c in zip(y, colors):
    #     plt.scatter(X_d[y == i, 0], X_d[y == i, 1], c=c, s=3)
    for i in range(err_list.shape[0]):
        plt.scatter(X_d[err_list[i], 0], X_d[err_list[i], 1], label='Example legend entry.', s=80, marker=r'o',
                    facecolors='none',
                    edgecolors='red', linewidths=1)
    ax.axis("off")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)   
    plt.savefig("/Users/busraozgode/Desktop/t-SNE/emb_mnist_10000.pdf", bbox_inches="tight")
    plt.show()
    
def create_subset(data, labels, size=50):
    np.random.seed(42)
    ind = np.random.randint(0, data.shape[0], size=size)
    subdata = data[ind]
    sublabels = labels[ind]
    return subdata, sublabels

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

X_emb=np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/mnist_test/X_emb_0.npy")
y=np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/mnist_test/y_test.npy")
x_test = np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/mnist_test/features_0.npy")

err=find_errors_majority(X_emb, y)

plot_embedding_with_errors(X_emb, y, err)

#%%
def plot_embedding_with_errors(X_d, y, err_list):
    labels = str(y).strip('[]')
    plt.figure()
    ax = plt.gca()
    colors = 'lightpink', 'g', 'b', 'orange', 'c', 'm', 'y', 'k', 'olive', 'purple', 'turquoise', 'lawngreen', 'gold', 'lime', 'coral' 
    for i in np.unique(y):
        plt.scatter(X_d[np.where(y == i), 0], X_d[np.where(y == i), 1], c=colors[i], s=5)
    # for i, c in zip(y, colors):
    #     plt.scatter(X_d[y == i, 0], X_d[y == i, 1], c=c, s=3)
    for i in range(err_list.shape[0]):
        plt.scatter(X_d[err_list[i], 0], X_d[err_list[i], 1], label='Example legend entry.', s=80, marker=r'o',
                    facecolors='none',
                    edgecolors='red', linewidths=1)
    ax.axis("off")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)   
    plt.savefig("/Users/busraozgode/Desktop/t-SNE/emb_AMB18_test.pdf", bbox_inches="tight")
    plt.show()
X_AMB=np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/AMB_integrated/X_test_emb.npy")
y_AMB=np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/AMB_integrated/y_test.npy")

err=find_errors_majority(X_AMB, y_AMB)
plot_embedding_with_errors(X_AMB,y_AMB,err)

#%%
def plot_embedding_with_errors(X_d, y, err_list):
    labels = str(y).strip('[]')
    plt.figure()
    ax = plt.gca()
    colors = 'lightpink', 'g', 'b', 'orange', 'c', 'm', 'y', 'k', 'olive', 'purple', 'turquoise', 'lawngreen', 'gold', 'lime', 'coral' 
    for i in np.unique(y):
        plt.scatter(X_d[np.where(y == i), 0], X_d[np.where(y == i), 1], c=colors[i], s=5)
    # for i, c in zip(y, colors):
    #     plt.scatter(X_d[y == i, 0], X_d[y == i, 1], c=c, s=3)
    for i in range(err_list.shape[0]):
        plt.scatter(X_d[err_list[i], 0], X_d[err_list[i], 1], label='Example legend entry.', s=80, marker=r'o',
                    facecolors='none',
                    edgecolors='red', linewidths=1)
    ax.axis("off")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)   
    plt.savefig("/Users/busraozgode/Desktop/t-SNE/emb_simulated.pdf", bbox_inches="tight")
    plt.show()
X_sim=np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/Simulated/X_emb_0.npy")
y_sim=np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/Simulated/y.npy")

err=find_errors_majority(X_sim, y_sim)
plot_embedding_with_errors(X_sim,y_sim,err)

#%%
def tsne_p(data, dim=2, perplexity=30):
    #X = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
    X_embedded = TSNE(n_components=dim, perplexity=perplexity).fit_transform(data)
    return X_embedded
def plot_embedding_with_errors(X_d, y, err_list):
    labels = str(y).strip('[]')
    plt.figure()
    ax = plt.gca()
    colors = 'lightpink', 'g', 'b', 'orange', 'c', 'm', 'y', 'k', 'olive', 'purple', 'turquoise', 'lawngreen', 'gold', 'lime', 'coral' 
    for i in np.unique(y):
        plt.scatter(X_d[np.where(y == i), 0], X_d[np.where(y == i), 1], c=colors[i], s=5)
    # for i, c in zip(y, colors):
    #     plt.scatter(X_d[y == i, 0], X_d[y == i, 1], c=c, s=3)
    for i in range(err_list.shape[0]):
        plt.scatter(X_d[err_list[i], 0], X_d[err_list[i], 1], label='Example legend entry.', s=80, marker=r'o',
                    facecolors='none',
                    edgecolors='red', linewidths=1)
    ax.axis("off")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)   
    plt.savefig("/Users/busraozgode/Desktop/t-SNE/emb_fashion_mnist_5000.pdf", bbox_inches="tight")
    plt.show()
X_fas=np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/fashion_mnist/X_emb_0.npy")
y_fas=np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/fashion_mnist/y.npy")

# err=find_errors_majority(X_fas, y_fas)
# plot_embedding_with_errors(X_fas,y_fas,err)

X_te2=create_subset(X_fas,y_fas,5000)
X_new2, y_new2=X_te2[0], X_te2[1]

X_emb2=tsne_p(X_new2)

err=find_errors_majority(X_emb2, y_new2)
plot_embedding_with_errors(X_emb2,y_new2,err)

#%%
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import glob
from pathlib import Path
import scipy.spatial.distance as dist

def plot_embedding_with_errors(X_d, y, err_list):
    labels = str(y).strip('[]')
    plt.figure()
    ax = plt.gca()
    colors = 'purple', 'g', 'b', 'orange', 'c', 'm', 'y', 'k', 'lightpink','olive'
    for i in np.unique(y):
        plt.scatter(X_d[np.where(y == i), 0], X_d[np.where(y == i), 1], c=colors[i], s=5)
    # for i, c in zip(y, colors):
    #     plt.scatter(X_d[y == i, 0], X_d[y == i, 1], c=c, s=3)
    for i in range(err_list.shape[0]):
        plt.scatter(X_d[err_list[i], 0], X_d[err_list[i], 1], label='Example legend entry.', s=80, marker=r'o',
                    facecolors='none',
                    edgecolors='red', linewidths=1)
    ax.axis("off")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)   
    plt.savefig("/Users/busraozgode/Desktop/t-SNE/mnist_after_cancel_th_0_5.pdf", bbox_inches="tight")
    plt.show()
    
def create_subset(data, labels, size=50):
    np.random.seed(42)
    ind = np.random.randint(0, data.shape[0], size=size)
    subdata = data[ind]
    sublabels = labels[ind]
    return subdata, sublabels

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

th=0.5 #threshold value for y_pred
X_emb=np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/mnist_test/X_emb_0.npy")
y=np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/mnist_test/y_test.npy")
model=pickle.load(open("/Users/busraozgode/Desktop/t-SNE/best_rf_model_sorted_500.sav", 'rb'))
x_test = np.load("/Users/busraozgode/Desktop/t-SNE/Datasets/mnist_test/features_0.npy")
X_test=np.reshape(x_test, [x_test.shape[0], x_test.shape[1]*x_test.shape[2]])

y_pred = model.predict(X_test)
ind = [ n for n, i in enumerate(y_pred) if i>th]

X_emb_del = X_emb[ind]
y_del = y [ind]
err = find_errors_majority(X_emb_del, y_del)
plot_embedding_with_errors(X_emb_del,y_del,err)




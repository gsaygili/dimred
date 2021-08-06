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
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'olive', 'orange', 'purple'
    for i in np.unique(y):
        plt.scatter(X_d[np.where(y == i), 0], X_d[np.where(y == i), 1], c=colors[i], s=3)
    # for i, c in zip(y, colors):
    #     plt.scatter(X_d[y == i, 0], X_d[y == i, 1], c=c, s=3)
    for i in range(err_list.shape[0]):
        plt.scatter(X_d[err_list[i], 0], X_d[err_list[i], 1], label='Example legend entry.', s=80, marker=r'o',
                    facecolors='none',
                    edgecolors='red')
    for i in range(corr_list.shape[0]):
        plt.scatter(X_d[corr_list[i], 0], X_d[corr_list[i], 1], label='Example legend entry.', s=80, marker=r'o',
                    facecolors='none',
                    edgecolors='green')
    plt.show()

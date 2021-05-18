import load_mnist as mnist
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import glob
from pathlib import Path


def tsne_p(data, dim=2, perplexity=30):
    X = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
    X_embedded = TSNE(n_components=dim, perplexity=perplexity).fit_transform(X)
    return X_embedded


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
        plt.scatter(X_d[err_list[i], 0], X_d[err_list[i], 1], label='Example legend entry.', s=80, marker=r'o', facecolors='none',
                edgecolors='red')
    plt.show()


def apply_tsne_subsets(dim=2, perplexity=30, size=5000, mnist_folder="/home/gorkem/datasets/"):
    _, _, save_path = mnist.mnist_subsets(mnist_folder=mnist_folder, size=size)
    # find number of subsets
    X_files = sorted(glob.glob(save_path+"X*.npy"))
    for i in range(len(X_files)):
        X = np.load(X_files[i])
        X_emb = tsne_p(X, dim=dim, perplexity=perplexity)
        # create an embedding subfolder and save with perplexity info
        Path(save_path+"emb_p"+str(perplexity)+"/").mkdir(parents=True, exist_ok=True)
        np.save(save_path+"emb_p"+str(perplexity)+"/"+"Xemb_"+str(i), X_emb)


emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
Xe = np.load(emb_folder+"Xemb_0.npy")
y = np.load(y_folder+"y_5000_0.npy")
# plot_embedding(Xe, y)



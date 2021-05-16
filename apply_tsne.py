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


def apply_tsne_subsets(dim=2, perplexity=30, size=5000, mnist_folder="/home/gorkem/datasets/"):
    _, _, save_path = mnist.mnist_subsets(mnist_folder=mnist_folder, size=size)
    # find number of subsets
    X_files = sorted(glob.glob(save_path+"X*.npy"))
    for i in range(len(X_files)):
        X = np.load(X_files[i])
        X_emb = tsne_p(X, dim=2, perplexity=perplexity)
        # create an embedding subfolder and save with perplexity info
        Path(save_path+"emb_p"+str(perplexity)+"/").mkdir(parents=True, exist_ok=True)
        np.save(save_path+"emb_p"+str(perplexity)+"/"+"Xemb_"+str(i), X_emb)


# plt.figure(figsize=(6, 5))
# colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
# for i, c in zip(y, colors):
#     plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], c=c)
# plt.show()
apply_tsne_subsets()

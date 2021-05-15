import load_mnist as data
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


# very important perplexity parameter
perplexity = 30

X, y = data.mnist_subset(size=5000)
X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
X_embedded = TSNE(n_components=2).fit_transform(X)

plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c in zip(y, colors):
    plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], c=c)
plt.show()

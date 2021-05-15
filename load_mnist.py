import torchvision.datasets as datasets
from pathlib import Path
import numpy as np


def create_subset(data, labels, size=50):
    np.random.seed(42)
    ind = np.random.randint(0, data.shape[0], size=size)
    subdata = data[ind]
    sublabels = labels[ind]
    return subdata, sublabels


mnist_folder = "/home/gorkem/datasets/"
Path(mnist_folder).mkdir(parents=True, exist_ok=True)

# download/load mnist training dataset
mnist_trainset = datasets.MNIST(root=mnist_folder, train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root=mnist_folder, train=False, download=True, transform=None)

X_train = mnist_trainset.data.numpy()
y_train = mnist_trainset.targets.numpy()
X_test = mnist_testset.data.numpy()
y_test = mnist_testset.data.numpy()

sX_tr, sy_tr = create_subset(X_train, y_train)

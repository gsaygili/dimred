import torchvision.datasets as datasets
from pathlib import Path
import numpy as np


def create_subset(data, labels, size=50):
    np.random.seed(42)
    ind = np.random.randint(0, data.shape[0], size=size)
    subdata = data[ind]
    sublabels = labels[ind]
    return subdata, sublabels


def create_subsets(data, labels, size=5000, is_save=True, save_path="/home/gorkem/datasets/mnist_subsets/"):
    save_path = save_path + str(size) + "/"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)
    data = data[inds]
    labels = labels[inds]
    X_all = []
    y_all = []
    for i in range(np.int16(np.floor(data.shape[0]/size))):
        if ((i+1) * size) <= data.shape[0]:
            X = data[size*i:size*(i+1)]
            y = labels[size*i:size*(i+1)]
            X_all.append(X)
            y_all.append(y)
            if is_save:
                np.save(save_path + "X_" + str(size) + "_" + str(i), X)
                np.save(save_path + "y_" + str(size) + "_" + str(i), y)
        else:
            X = data[size*i:]
            y = labels[size*i:]
            X_all.append(X)
            y_all.append(y)
            if is_save:
                np.save(save_path + "X_" + str(size) + "_" + str(i), X)
                np.save(save_path + "y_" + str(size) + "_" + str(i), y)
    if is_save:
        return X_all, y_all, save_path
    else:
        return X_all, y_all


# run this function to create just one random subset
def mnist_subset(mnist_folder="/home/gorkem/datasets/", size=5000, is_train=True):
    Path(mnist_folder).mkdir(parents=True, exist_ok=True)

    # download/load mnist training dataset
    if is_train:
        mnist_trainset = datasets.MNIST(root=mnist_folder, train=True, download=True, transform=None)
        X_train = mnist_trainset.data.numpy()
        y_train = mnist_trainset.targets.numpy()
        sX, sy = create_subset(X_train, y_train, size=size)
    else:
        mnist_testset = datasets.MNIST(root=mnist_folder, train=False, download=True, transform=None)
        X_test = mnist_testset.data.numpy()
        y_test = mnist_testset.targets.numpy()
        sX, sy = create_subset(X_test, y_test, size=size)
    return sX, sy


# Run this function to create subsets
def mnist_subsets(mnist_folder="/home/gorkem/datasets/", size=5000, is_train=True, is_save=True):
    Path(mnist_folder).mkdir(parents=True, exist_ok=True)

    # download/load mnist training dataset
    if is_train:
        mnist_trainset = datasets.MNIST(root=mnist_folder, train=True, download=True, transform=None)
        X_train = mnist_trainset.data.numpy()
        y_train = mnist_trainset.targets.numpy()
        return create_subsets(data=X_train, labels=y_train, size=size, is_save=is_save,
                              save_path=mnist_folder+'mnist_subsets/')
    else:
        mnist_testset = datasets.MNIST(root=mnist_folder, train=False, download=True, transform=None)
        X_test = mnist_testset.data.numpy()
        y_test = mnist_testset.targets.numpy()
        return create_subsets(data=X_test, labels=y_test, size=size, is_save=is_save,
                              save_path=mnist_folder+'mnist_subsets/')

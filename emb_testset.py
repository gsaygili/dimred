# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:21:22 2021

@author: Admin
"""
from apply_tsne import tsne_p
from sklearn.manifold import TSNE
from pathlib import Path
import numpy as np
from keras.datasets import mnist
import torchvision.datasets as datasets

def apply_tsne_testset(perplexity=30):
    save_path = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/"
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    X_test = mnist_testset.data.numpy()
    np.save(save_path+"emb_p"+str(perplexity)+"/"+"X_test", X_test)
    y_test = mnist_testset.targets.numpy()
    X_te_emb = tsne_p(X_test)
    # create an embedding subfolder and save with perplexity info
    Path(save_path+"emb_p"+str(perplexity)+"/").mkdir(parents=True, exist_ok=True)
    np.save(save_path+"emb_p"+str(perplexity)+"/"+"X_te_emb_", X_te_emb)
    np.save(save_path+"emb_p"+str(perplexity)+"/"+"y_test", y_test)
    return X_te_emb, y_test
X_te_emb,y_test=apply_tsne_testset()
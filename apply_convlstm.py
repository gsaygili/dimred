# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:02:04 2021

@author: Admin
"""

import calc_error as err
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers import LSTM, ConvLSTM2D, MaxPooling2D
import keras
from keras import optimizers
import scipy.io as sio
from sys import platform
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys


def evaluate_model_convlstm(X_train, y_train, X_test, y_test, n_features):
    InpShape=(None, x_train.shape[1], x_train.shape[2], 1)
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=InpShape, padding='same',activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='tanh',padding='same',return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh',padding='same',return_sequences=False))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense( units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics = ['accuracy',keras.metrics.SpecificityAtSensitivity(0.8)])
    bsize = 8
    history = model.fit(X_train, y_train, batch_size=bsize, validation_split=0.1, epochs=30)
    y_predict = model.predict(X_test, batch_size=bsize)  
#    yhat = []
#    for pred in y_predict:
#        if pred >= 0.8:
#            yhat.append(1)
#        else:
#            yhat.append(0)        
    yhat = model.predict_classes(X_test, batch_size=bsize)
    score = model.evaluate(X_test, y_test, batch_size=bsize)
    print("Accuracy: %.2f%%" % (score[1] * 100))
    return score[1], y_predict, model, yhat, history, model

if platform == "linux" or platform == "linux2":
    emb_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/"
elif platform == "darwin":
    emb_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/"
elif platform == "win32":
    emb_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/"


# test parameters
train_id = 2
test_id = 3
B = 0

X_tr = np.load(emb_folder + "Xemb_"+str(train_id)+".npy")
y_tr = np.load(y_folder + "y_5000_"+str(train_id)+".npy")

X_te = np.load(emb_folder + "Xemb_"+str(test_id)+".npy")
y_te = np.load(y_folder + "y_5000_"+str(test_id)+".npy")

if B == 0:
    x_train = np.load(emb_folder + "features_" + str(train_id) + ".npy")
    x_test = np.load(emb_folder + "features_" + str(test_id) + ".npy")
elif B == 5 or B == 10 or B == 20:
    x_train = np.load(emb_folder + "avg_features_" + str(train_id) + "_blocksize_" + str(B) + ".npy")
    x_test = np.load(emb_folder + "avg_features_" + str(test_id) + "_blocksize_" + str(B) + ".npy")
else:
    sys.exit('invalid block size')

y_tr_ind = err.find_errors_majority(X_tr, y_tr)
y_train = np.zeros(X_tr.shape[0])
y_train[y_tr_ind] = 1

y_te_ind = err.find_errors_majority(X_te, y_te)
y_test = np.zeros(X_te.shape[0])
y_test[y_te_ind] = 1

x_train_cl = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2], 1)
x_test_cl = x_test.reshape(x_test.shape[0], 1,  x_test.shape[1], x_test.shape[2], 1)

acc, y_predict, model, y_label, history, model = evaluate_model_convlstm(x_train_cl, y_train, x_test_cl, y_test, x_train.shape[2])
model.save(emb_folder+"convlstm_model_"+str(train_id))
print("accuracy: ", acc)
print('accuracy: ', accuracy_score(y_test, y_label))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# ------------------------------
#plot roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)
print('Area under the ROC curve : %f' % roc_auc)

plt.figure(0)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='(AUC = %0.2f, Acc = %0.2f)' % (roc_auc, acc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(emb_folder+'1-roc_curve_tr_'+str(train_id)+'_te_'+str(test_id)+'_bs_'+str(B)+'.png')
plt.show()


# ------------------------------
# plot confusion matrix
C = confusion_matrix(y_test, y_label)
plt.figure()
df_cm = pd.DataFrame(C, range(2), range(2))
ax= plt.subplot()
sns.set(font_scale=1.4)  # for label size
seaplt = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt='g', ax=ax)  # font size
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['true', 'false'],ha='center'); ax.yaxis.set_ticklabels(['true', 'false'], ha='center');
splt = seaplt.get_figure()
splt.savefig(emb_folder+'2-confusion_matrix_tr_'+str(train_id)+'_te_'+str(test_id)+'_bs_'+str(B)+'.png')
tn, fp, fn, tp = confusion_matrix(y_test, y_label).ravel()
print('TP: ', tp)
print('TN: ', tn)
print('FP: ', fp)
print('FN: ', fn)

#%%
#import scipy.spatial.distance as dist
#def plot_embedding_with_errors(X_d, y, err_list):
##    labels = str(y).strip('[]')
#    plt.figure()
#    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'olive', 'orange', 'purple'
#    for i in np.unique(y):
#        plt.scatter(X_d[np.where(y == i), 0], X_d[np.where(y == i), 1], c=colors[i], s=3)
#    # for i, c in zip(y, colors):
#    #     plt.scatter(X_d[y == i, 0], X_d[y == i, 1], c=c, s=3)
#    for i in range(err_list.shape[0]):
#        plt.scatter(X_d[err_list[i], 0], X_d[err_list[i], 1], label='Example legend entry.', s=80, marker=r'o',
#                    facecolors='none',
#                    edgecolors='red')
#    plt.savefig(emb_folder+'3-embedding_map_with_errors_'+str(train_id)+'_te_'+str(test_id)+'_bs_'+str(B)+'.png')
#    plt.show()
#    
#def find_errors_majority(X_emb, labels, K=20):
#    # take a closest neighborhood of K and check whether the label of the majority is the same as the center
#    X_d = dist.squareform(dist.pdist(X_emb, "euclidean"))
#    # the first column is the sample itself since the distance is zero
#    sort_index = np.argsort(X_d)
#    error_list = []
#    for i in range(X_d.shape[0]):
#        K_neigh = sort_index[i, 1:K+1]
#        lab_neigh = labels[K_neigh]
#        num_label = np.count_nonzero(lab_neigh == labels[i])
#        if num_label < K/2:
#            error_list.append(i)
#    err = np.array(error_list)
#    return err
#
#if platform == "linux" or platform == "linux2":
#    emb_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/emb_p30/"
#    y_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/"
#elif platform == "darwin":
#    emb_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/emb_p30/"
#    y_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/"
#elif platform == "win32":
#    emb_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/emb_p30/"
#    y_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/"
#    
#Xe = np.load(emb_folder+"Xemb_1.npy")
#y = np.load(y_folder+"y_5000_1.npy")
#
#
#
#sort_index = np.argsort(y_predict)
#error_list = []
#for i in range(y_predict.shape[0]):
#    if y_label[i]==1 and y_test[i]==1:
#        error_list.append(i)
#err = np.array(error_list)
#
#err2=find_errors_majority(Xe, y)
#
#plot_embedding_with_errors(Xe, y, err)
#
#plot_embedding_with_errors(Xe, y, err2)
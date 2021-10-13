# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:51:13 2021

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
from extract_features import extract_feats
import scipy.spatial.distance as dist
import time
from sys import platform

if platform == "linux" or platform == "linux2":
    emb_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/"
elif platform == "darwin":
    emb_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/"
elif platform == "win32":
    emb_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/5000/"


X_train = []
y_train = []
for train_id in range(12):

    X_tr = np.load(emb_folder + "Xemb_"+str(train_id)+".npy")
    y_tr = np.load(y_folder + "y_5000_"+str(train_id)+".npy") 
    x_train = np.load(emb_folder + "features_" + str(train_id) + ".npy")
    y_tr_ind = err.find_errors_majority(X_tr, y_tr)
    y_tra = np.zeros(X_tr.shape[0])
    y_tra[y_tr_ind] = 1
    X_train.append(x_train)
    y_train.append(y_tra)

X_train2 = np.array(X_train)
y_train2= np.array(y_train)
X_train2=np.reshape(X_train2, [X_train2.shape[0]*X_train2.shape[1],1,X_train2.shape[2],X_train2.shape[3],1])
y_train2=np.reshape(y_train2, [y_train2.shape[0]*y_train2.shape[1],1])


#%%
emb_folder="C:/Users/Admin/Documents/GitHub/dimred/datasets/mnist_subsets/emb_p30/"
X_test = np.load(emb_folder + "X_te_emb_.npy")
y_te = np.load(emb_folder + "y_test.npy")
x_te_emb = np.load(emb_folder + "features_0.npy")

y_te_ind = err.find_errors_majority(X_test, y_te)
y_test = np.zeros(X_test.shape[0])
y_test[y_te_ind] = 1

X_test=np.reshape(x_te_emb, [x_te_emb.shape[0], 1, x_te_emb.shape[1], x_te_emb.shape[2], 1])
y_test=np.reshape(y_test, [y_test.shape[0],1])

#%%
def evaluate_model_convlstm(X_train, y_train, X_test, y_test, n_features):
    InpShape=(None, 20, 6, 1)
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=InpShape, padding='same',activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='tanh',padding='same',return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh',padding='same',return_sequences=False))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense( units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics = [keras.metrics.SpecificityAtSensitivity(0.5)])
    bsize = 64
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

acc, y_predict, model, y_label, history, model = evaluate_model_convlstm(X_train2, y_train2, X_test, y_test, X_train2.shape[2])
model.save(emb_folder+"convlstm_model_all_training_set_SpecificityAtSensitivity")
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
plt.savefig(emb_folder+'ROC Curve_all training set_SpecificityAtSensitivity.png')
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
splt.savefig(emb_folder+'Confusion matrix_all training set_SpecificityAtSensitivity.png')
tn, fp, fn, tp = confusion_matrix(y_test, y_label).ravel()
print('TP: ', tp)
print('TN: ', tn)
print('FP: ', fp)
print('FN: ', fn)
#%%
import os
os.chdir(emb_folder)
os.mkdir(emb_folder+"results_SpecificityAtSensitivity/")
os.chdir(emb_folder+"results_SpecificityAtSensitivity/")
np.save("y_predict.npy", y_predict)
np.save("y_label.npy", y_label)
np.save("fpr.npy", fpr)
np.save("tpr.npy", tpr)
np.save("roc.npy",roc_auc)
np.save("acc.npy",acc)
np.save("cm.npy",df_cm)

sensitivity = C[0,0]/(C[0,0]+C[0,1])
print('Sensitivity : ', sensitivity)

specificity = C[1,1]/(C[1,0]+C[1,1])
print('Specificity : ', specificity)
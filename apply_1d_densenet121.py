# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 22:49:26 2021

@author: Admin
"""
from densenet.classifiers.one_d import DenseNet121

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

def evaluate_model_densenet121(X_train, y_train, X_test, y_test):
    
    model = DenseNet121(input_shape=(20, 6),num_outputs=1)  
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics = ['accuracy'])
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
train_id = 0
test_id = 1
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

#x_train_cl = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])
#x_test_cl = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2])

acc, y_predict, model, y_label, history, model = evaluate_model_densenet121(x_train, y_train, x_test, y_test)
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


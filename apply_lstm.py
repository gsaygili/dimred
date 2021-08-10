import calc_error as err
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras import optimizers
import scipy.io as sio
from sys import platform
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def evaluate_model_lstm(X_train, y_train, X_test, y_test, n_features):
    model = Sequential()
    model.add(LSTM(32,  activation='tanh', return_sequences=True, input_shape=(None, n_features)))
    model.add(BatchNormalization())
    model.add(LSTM(16, activation='tanh', return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = optimizers.Adam(clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    bsize = 16
    model.fit(X_train, y_train, validation_split=0.2, batch_size=bsize, epochs=20)
    y_predict = model.predict(X_test, batch_size=bsize)
    y_prdlabel = model.predict_classes(X_test,batch_size=bsize)
    score = model.evaluate(X_test, y_test, batch_size=bsize)
    print("Accuracy: %.2f%%" % (score[1] * 100))
    return score[1], y_predict, model, y_prdlabel


if platform == "linux" or platform == "linux2":
    emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
elif platform == "darwin":
    emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
elif platform == "win32":
    emb_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/"


train_id = 0
X_tr = np.load(emb_folder + "Xemb_"+str(train_id)+".npy")
x_train = np.load(emb_folder + "features_"+str(train_id)+".npy")
y_tr = np.load(y_folder + "y_5000_"+str(train_id)+".npy")

test_id = 11
X_te = np.load(emb_folder + "Xemb_"+str(test_id)+".npy")
x_test = np.load(emb_folder + "features_"+str(test_id)+".npy")
y_te = np.load(y_folder + "y_5000_"+str(test_id)+".npy")

y_tr_ind = err.find_errors_majority(X_tr, y_tr)
y_train = np.zeros(X_tr.shape[0])
y_train[y_tr_ind] = 1

y_te_ind = err.find_errors_majority(X_te, y_te)
y_test = np.zeros(X_te.shape[0])
y_test[y_te_ind] = 1

acc, y_predict, model, y_label = evaluate_model_lstm(x_train, y_train, x_test, y_test, x_train.shape[2])
print("accuracy: ", acc)
print('accuracy: ', accuracy_score(y_test, y_label))


# ------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)
print('Area under the ROC curve : %f' % roc_auc)

plt.figure(0)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig('roc_curve.pdf')

# ------------------------------
# plot confusion matrix
C = confusion_matrix(y_test, y_label)
plt.figure()
df_cm = pd.DataFrame(C, range(2), range(2))
sns.set(font_scale=1.4)  # for label size
seaplt = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
splt = seaplt.get_figure()
splt.savefig('confusion_matrix.pdf')
tn, fp, fn, tp = confusion_matrix(y_test, y_label).ravel()
print('TP: ', tp)
print('TN: ', tn)
print('FP: ', fp)
print('FN: ', fn)



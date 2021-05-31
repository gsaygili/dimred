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


def evaluate_model_lstm(X_train, y_train, X_test, y_test, n_features):
    model = Sequential()
    model.add(LSTM(16,  activation='tanh', return_sequences=True, input_shape=(None, n_features)))
    model.add(BatchNormalization())
    model.add(LSTM(16, activation='tanh', return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = optimizers.Adam(clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    bsize = 64
    model.fit(X_train, y_train, validation_split=0.2, batch_size=bsize, epochs=20)
    y_predict = model.predict(X_test, batch_size=bsize)
    y_prdlabel = model.predict_classes(X_test,batch_size=bsize)
    score = model.evaluate(X_test, y_test, batch_size=bsize)
    print("Accuracy: %.2f%%" % (score[1] * 100))
    return score[1], y_predict, model, y_prdlabel


emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"

X_tr = np.load(emb_folder + "Xemb_0.npy")
x_train = np.load(emb_folder + "features_euc_corr_0.npy")
y_tr = np.load(y_folder + "y_5000_0.npy")

X_te = np.load(emb_folder + "Xemb_1.npy")
x_test = np.load(emb_folder + "features_euc_corr_1.npy")
y_te = np.load(y_folder + "y_5000_1.npy")

y_tr_ind = err.find_errors_majority(X_tr, y_tr)
y_train = np.zeros(X_tr.shape[0])
y_train[y_tr_ind] = 1

y_te_ind = err.find_errors_majority(X_te, y_te)
y_test = np.zeros(X_te.shape[0])
y_test[y_te_ind] = 1

acc, y_predict, model, y_label = evaluate_model_lstm(x_train, y_train, x_test, y_test, 2)
print("accuracy: ", acc)


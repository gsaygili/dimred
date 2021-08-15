import numpy as np
from sys import platform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sys
import calc_error as err
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import pandas as pd
import pickle

sensitivity = make_scorer(recall_score, pos_label=1)
param_grid = {
    'n_estimators': [10, 20, 50, 100, 200],
    'max_depth': [2, 3, 5, 10, 20],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 3, 5, 10, 100],
}

if platform == "linux" or platform == "linux2":
    emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
elif platform == "darwin":
    emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
elif platform == "win32":
    emb_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/"

# test parameters
train_id = 0
test_id = 1
B = 20

X_tr = np.load(emb_folder + "Xemb_"+str(train_id)+".npy")
y_tr = np.load(y_folder + "y_5000_"+str(train_id)+".npy")

X_te = np.load(emb_folder + "Xemb_"+str(test_id)+".npy")
y_te = np.load(y_folder + "y_5000_"+str(test_id)+".npy")

if B == 0:
    x_train_ = np.load(emb_folder + "features_" + str(train_id) + ".npy")
    x_test_ = np.load(emb_folder + "features_" + str(test_id) + ".npy")
elif B == 5 or B == 10 or B == 20:
    x_train_ = np.load(emb_folder + "avg_features_" + str(train_id) + "_blocksize_" + str(B) + ".npy")
    x_test_ = np.load(emb_folder + "avg_features_" + str(test_id) + "_blocksize_" + str(B) + ".npy")
else:
    sys.exit('invalid block size')
x_train = np.reshape(x_train_, (x_train_.shape[0], x_train_.shape[1]*x_train_.shape[2]))
x_test = np.reshape(x_test_, (x_test_.shape[0], x_test_.shape[1]*x_test_.shape[2]))

y_tr_ind = err.find_errors_majority(X_tr, y_tr)
y_train = np.zeros(X_tr.shape[0])
y_train[y_tr_ind] = 1

y_te_ind = err.find_errors_majority(X_te, y_te)
y_test = np.zeros(X_te.shape[0])
y_test[y_te_ind] = 1

grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                    param_grid, refit=True, verbose=0, cv=3,
                    scoring=sensitivity, n_jobs=-1)
grid.fit(x_train, y_train)
clf = grid.best_estimator_
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)

filename = emb_folder + "rf_model_" + str(train_id) + ".sav"
pickle.dump(clf, open(filename, 'wb'))

# ------------------------------
acc = accuracy_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
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
plt.show()
plt.savefig(emb_folder+'1_RF_roc_curve_tr_'+str(train_id)+'_te_'+str(test_id)+'_bs_'+str(B)+'.png')

# ------------------------------
# plot confusion matrix
C = confusion_matrix(y_test, y_pred)
plt.figure()
df_cm = pd.DataFrame(C, range(2), range(2))
sns.set(font_scale=1.4)  # for label size
seaplt = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
splt = seaplt.get_figure()
splt.savefig(emb_folder+'2_RF_confusion_matrix_tr_'+str(train_id)+'_te_'+str(test_id)+'_bs_'+str(B)+'.png')
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('TP: ', tp)
print('TN: ', tn)
print('FP: ', fp)
print('FN: ', fn)

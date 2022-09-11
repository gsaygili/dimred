
#
#  apply_RF.py
#
# Implementation of confidence estimation algorithm in Python. 
# The implementation was tested on Python 3.7. 
# In order to plot the graphs, a working installation of matplotlib is required.
# The confidence prediction algorithm can be run by executing: `ipython apply_RF.py` by specifying the required inputs of the algorithm.
#
# 
# This framework implement the following steps, respectively:
# 
#  - Extract the features from the original and embedding spaces of the dataset according to 6 distance measures ("euclidean", "cosine", "correlation", "chebyshev", "canberra", "braycurtis" )
#  - Calculate the ground truth confident scores from the training set using the embedding and labels that will be used as an input for the RF regression algorithm (the model learns the confidence scores in supervised manner)
#  - Run the RF regressor on training set by using gridsearch to get the optimized model.
#  - Calculate the confidence scores for test set using the best RF model obtained.
#

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
from extract_features import extract_feats
from calc_error import find_error_score, calc_npr
from evaluate_model import evaluate_regression
from apply_tsne import tsne_p
import os 

def confidence_prediction(X_train, X_train_emb, X_test, X_test_emb, y_train):
    
    X_train_feat = extract_feats(X_train, X_train_emb)
    X_test_feat = extract_feats(X_test, X_test_emb)
    
    #reshaping feature data
    X_train_feat = np.reshape(X_train_feat, [X_train_feat.shape[0], X_train_feat.shape[1]*X_train_feat.shape[2]])
    X_test_feat = np.reshape(X_test_feat, [X_test_feat.shape[0], X_test_feat.shape[1]*X_test_feat.shape[2]])
    
    #calculate error scores for training set (y_tr_score)
    y_tr_ind, y_tr_score = find_error_score(X_train_emb, y_train) # y_tr_ind--indexes of erroneous samples
    y_tra = np.ones(X_train_emb.shape[0]) # y_tra--binary labels (erroneous-0,corrects-1) 
    y_tra[y_tr_ind] = 0
    y_tr_score= np.array(y_tr_score)

    #Train the model
    print("training of the model")

    param_grid = {
        'n_estimators': [20, 50, 100, 200],
        'max_features': ["auto", "sqrt"],
        'criterion': ["squared_error"],
        'max_depth': [2, 5, 10, 20],
        'min_samples_split': [2, 5, 10, 20],
    }

    grid = GridSearchCV(RandomForestRegressor(random_state=42),
                    param_grid, refit=True, verbose=0, cv=3)
    grid.fit(X_train_feat, y_tr_score)

    #test on best model
    clf = grid.best_estimator_
    clf.fit(X_train_feat, y_tr_score)
    pred_conf_score = clf.predict(X_test_feat)

    #save model
    pickle.dump(clf, open("best_RF_model.sav", 'wb'))
    print('completed')

    return pred_conf_score


if __name__ == "__main__":
    
    X_train = np.load('../X_test.npy') #original train set
    X_test = np.load('../X_train.npy') #original test set
    y_train = np.load('../y_train.npy', allow_pickle=True) # label_encoded_target_values for train set

    #calculation of the t-SNE
    X_train_emb = tsne_p(X_train, dim=2, perplexity=30)
    X_test_emb = tsne_p(X_test, dim=2, perplexity=30)

    #calculate confidence scores
    predicted_confidence_scores = confidence_prediction(X_train, X_train_emb, X_test, X_test_emb, y_train)

    



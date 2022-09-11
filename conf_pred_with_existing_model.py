#
#  conf_pred_with_existing_model.py
#
# This implementation is designed in order to use pre-trained model to get the confidence scores.
# Models trained on two different datasets called AMB18 and MNIST have been saved in this repository.
# One can use these pre-trained models to estimate confidence score on similar domains.
# Implementation of confidence estimation algorithm in Python. 
# The implementation was tested on Python 3.7. 
# plotting the evaluation results is optional can be used by evaluate_regression function. 
# In order to plot the graphs, a working installation of matplotlib is required.
# The confidence prediction algorithm can be run by executing: `ipython conf_pred_with_existing_model.py` by specifying the required inputs of the algorithm.
#
# 
# This framework implement the following steps, respectively:
# 
#  - Extract the features from the original and embedding spaces of the dataset according to 6 distance measures ("euclidean", "cosine", "correlation", "chebyshev", "canberra", "braycurtis" )
#  - Calculate the ground truth confident scores from the training set using the embedding and labels that will be used as an input for the RF regression algorithm (the model learns the confidence scores in supervised manner)
#  - Calculate the confidence scores for the dataset using the pre-trained RF model (the model trained on 'AMB18' or 'MNIST' can be chosen)
#  - Evaluate the performance of the model on test set.
#

import numpy as np
import pickle
from extract_features import extract_feats
from calc_error import find_error_score, calc_npr
from evaluate_model import evaluate_regression
from apply_tsne import tsne_p
import os

def confidence_prediction_with_existing_model(X, X_emb, y):
    # This function 
    X_feat = extract_feats(X, X_emb)

    #reshaping feature data
    X_feat = np.reshape(X_feat, [X_feat.shape[0], X_feat.shape[1]*X_feat.shape[2]])

    #calculate error scores for test set 
    y_ind, y_score = find_error_score(X_emb, y) # y_ind--indexes of erroneous samples
    y_tra = np.ones(X_emb.shape[0]) # y_tra--binary labels (erroneous-0, corrects-1) 
    y_tra[y_ind] = 0
    y_score= np.array(y_score)

    #calculate NPR for test set
    y_npr = calc_npr(X, X_emb)

    #EVALUATION PART (optional part)
    # the model trained on 'AMB18' or 'mnist' can be chosen
    get_model = "best_rf_model_for_AMB18.sav"
#    get_model = "best_rf_model_for_mnist.sav"

    # give the directory that you want to save your evaluation results
    save_folder = '/Results_of_the_model/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    clf = pickle.load(open(get_model, 'rb'))
    pred_conf_score = clf.predict(X_feat)
    evaluate_regression(clf, X_feat, y_score, y_ind, y_npr, save_folder)
    
    return pred_conf_score
if __name__ == "__main__":
    X = np.load('../X_train.npy') # original dataset
    y = np.load('../y_train.npy') # label_encoded_target_values 

    #calculation of the t-SNE
    X_emb = tsne_p(X, dim=2, perplexity=30)

    #calculate confidence scores
    predicted_confidence_scores = confidence_prediction_with_existing_model(X, X_emb, y)

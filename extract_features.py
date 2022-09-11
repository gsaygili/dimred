
import numpy as np
import scipy.spatial.distance as dist
import time

def extract_feats(X, X_emb, K=20, sample_id=0,
                  distance_measures=["euclidean", "cosine", "correlation", "chebyshev", "canberra", "braycurtis"]):

    features = np.zeros((X.shape[0], K, len(distance_measures)))
    X_d = dist.squareform(dist.pdist(X_emb, "euclidean"))
    sort_index_d = np.argsort(X_d)

    for ind, c in enumerate(distance_measures):
        print("Calculating " + c + " Distances")
        start_time = time.time()
        X_D = dist.squareform(dist.pdist(X, c))
        X_D = np.nan_to_num(X_D)
        X_D = (X_D - np.min(X_D))/np.ptp(X_D)
        sort_D = np.sort(X_D)
        for i in range(X_D.shape[0]):
            cost = X_D[i, :]
            s_D = sort_D[i, 1:K + 1]
            s_d = np.sort(cost[sort_index_d[i, 1:K + 1]])
            features[i, :, ind] = np.abs(s_D - s_d)

        print(np.any(np.isnan(features)))
        print("--- " + c + " takes: %s seconds ---" % (time.time() - start_time))
    return(features)

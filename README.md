# Confidence estimation for t‑SNE embeddings using random forest
## B. Ozgode Yigin and Gorkem Saygili

Abstract:
Dimensionality reduction algorithms are commonly used for reducing the dimension of multi-dimensional data to visualize them on a standard display. Although many dimensionality reduction algorithms such as the t-distributed Stochastic Neighborhood Embedding aim to preserve close neighborhoods in low-dimensional space, they might not accomplish that for every sample of the data and eventually produce erroneous representations. In this study, we developed a supervised confidence estimation algorithm for detecting erroneous samples in embeddings. Our algorithm generates a confidence score for each sample in an embedding based on a distance-oriented score and a random forest regressor. We evaluate its performance on both intra- and inter-domain data and compare it with the neighborhood preservation ratio as our baseline. Our results showed that the resulting confidence score provides distinctive information about the correctness of any sample in an embedding compared to the baseline.

This code is the code of our journal publication:

***[1] B. Ozgode Yigin and G. Saygili, "Confidence estimation for t‑SNE embeddings using random forest", International Journal of Machine Learning and Cybernetics, 2022.***

Online available at:
https://link.springer.com/epdf/10.1007/s13042-022-01635-2?sharing_token=WHF414GgNmjoADmQasLa7ve4RwlQNchNByi7wbcMAY6tDVkBbSh45DjuKj43hFV3qga3b1UQE3Pb40D4zTiNcmW-0XY48mK9eedXGpzQbnRQ2y9SzJ9XZy8ZR0Z1JFgVtRhfTcs2HrmxHLausl2NjiPB9Y-igogtNeoT0-xTmV8%3D

***Please cite our paper [1] in case you use the code.***

Created by Busra Ozgode Yigin and Gorkem Saygili on 11-09-22.
Copyright (c) 2022 Tilburg University. All rights reserved.

Datasets:
1) MNIST
2) https://zenodo.org/record/4557712#.YUbplLgzZPY (AMB_integrated.zip)

How to use:
- You can run conf_pred_with_existing_model function for using pre-trained existing models on AMB18 and MNIST dataset on your test set.
- You can run conf_pred_with_training function for training your own model on your own training set and make confidence predictions on your test set.

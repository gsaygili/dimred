# dimred

Error/confidence estimation for (in particular) t-SNE.

-> For Unix platforms, you can use create_env.sh to create a virtual environment 
automatically. For other platforms, if you have anaconda3 installed in your system:

>> conda env create -f dimred_env.yml

A virtual environment called 'dimred' will be created automatically with required libraries.
Then use:

>> conda activate dimred

Everything is ready to go!

Datasets:
1) MNIST
2) https://zenodo.org/record/4557712#.YUbplLgzZPY (AMB_integrated.zip)

-> use Neighborhood Preservation Ratio (NPR)[1] as the target score to predict:
NPR: "For each word i, we measure the ratio of k similar words in the association data" [1]
-> choose k as 20.

[1] Van der Maaten, Laurens, and Geoffrey Hinton. "Visualizing non-metric similarities in multiple maps." Machine 
learning 87.1 (2012): 33-55.

import matplotlib.pyplot as plt
import numpy as np
import calc_error as cerr
import plot_embedding as plt_emb


from sys import platform
if platform == "linux" or platform == "linux2":
    emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
elif platform == "darwin":
    emb_folder = "/home/gorkem/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "/home/gorkem/datasets/mnist_subsets/5000/"
elif platform == "win32":
    emb_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/emb_p30/"
    y_folder = "C:/Users/gsayg/Dropbox/datasets/mnist_subsets/5000/"


# plot the n best embbedded samples and n worst embedded samples
sample_id = 2
X_embd = np.load(emb_folder + "Xemb_"+str(sample_id)+".npy")
x_cost = np.load(emb_folder + "features_"+str(sample_id)+".npy")
y_labl = np.load(y_folder + "y_5000_"+str(sample_id)+".npy")

N = 10
worstN, bestN = cerr.find_worst_best_N(X_embd, y_labl, K=20, N=N)
plt_emb.plot_embedding_with_errors_and_corrects(X_embd, y_labl, worstN, bestN)


# plot cost spaces
for f in range(x_cost.shape[2]):
    fig, axs = plt.subplots(N, 2)
    axs[0, 0].set_title('Worst Cost id:' + str(f))
    axs[0, 1].set_title('Best Cost id:' + str(f))
    for i in range(N):
        wid = worstN[i]
        bid = bestN[i]
        axs[i, 0].plot(x_cost[wid, :, f])
        axs[i, 1].plot(x_cost[bid, :, f])


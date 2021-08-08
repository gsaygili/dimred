import matplotlib.pyplot as plt
import numpy as np
from sys import platform
import calc_error as cerr
import plot_embedding as plt_emb


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
X_embd = np.load(emb_folder + "Xemb_0.npy")
x_cost = np.load(emb_folder + "features_0.npy")
y_labl = np.load(y_folder + "y_5000_0.npy")

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

# for i in range(N):
#     wid = worstN[i]
#     bid = bestN[i]
#     axs[0, 0].set_title('Worst Euc')
#     axs[0, 1].set_title('Best Euc')
#     for j in range(4):
#         axs[i, 0].plot(x_cost[wid, :, 0])
#         axs[i, 1].plot(x_cost[bid, :, 0])
#
# # plot cost spaces
# fig, axs = plt.subplots(N, 2)
# for i in range(N):
#     wid = worstN[i]
#     bid = bestN[i]
#     axs[0, 0].set_title('Worst SEuc')
#     axs[0, 1].set_title('Best SEuc')
#     for j in range(4):
#         axs[i, 0].plot(x_cost[wid, :, 1])
#         axs[i, 1].plot(x_cost[bid, :, 1])
#
# fig, axs = plt.subplots(N, 2)
# for i in range(N):
#     wid = worstN[i]
#     bid = bestN[i]
#     axs[0, 0].set_title('Worst Cosine')
#     axs[0, 1].set_title('Best Cosine')
#     for j in range(4):
#         axs[i, 0].plot(x_cost[wid, :, 2])
#         axs[i, 1].plot(x_cost[bid, :, 2])
#
# fig, axs = plt.subplots(N, 2)
# for i in range(N):
#     wid = worstN[i]
#     bid = bestN[i]
#     axs[0, 0].set_title('Worst correlation')
#     axs[0, 1].set_title('Best correlation')
#     for j in range(4):
#         axs[i, 0].plot(x_cost[wid, :, 3])
#         axs[i, 1].plot(x_cost[bid, :, 3])
#
# fig, axs = plt.subplots(N, 2)
# for i in range(N):
#     wid = worstN[i]
#     bid = bestN[i]
#     axs[0, 0].set_title('Worst chebyshev')
#     axs[0, 1].set_title('Best chebyshev')
#     for j in range(4):
#         axs[i, 0].plot(x_cost[wid, :, 4])
#         axs[i, 1].plot(x_cost[bid, :, 4])
#
# fig, axs = plt.subplots(N, 2)
# for i in range(N):
#     wid = worstN[i]
#     bid = bestN[i]
#     axs[0, 0].set_title('Worst canberra')
#     axs[0, 1].set_title('Best canberra')
#     for j in range(4):
#         axs[i, 0].plot(x_cost[wid, :, 5])
#         axs[i, 1].plot(x_cost[bid, :, 5])
#
# fig, axs = plt.subplots(N, 2)
# for i in range(N):
#     wid = worstN[i]
#     bid = bestN[i]
#     axs[0, 0].set_title('Worst braycurtis')
#     axs[0, 1].set_title('Best braycurtis')
#     for j in range(4):
#         axs[i, 0].plot(x_cost[wid, :, 6])
#         axs[i, 1].plot(x_cost[bid, :, 6])


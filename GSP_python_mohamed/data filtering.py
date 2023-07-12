import pickle
import numpy as np

import sys
from GSP_helper import cleanup, runGsp
from matplotlib import pyplot as plt

from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


inputs_list = ["N1", "P0", "T0", "Mach", "HP"]
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2", "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19"]

GEnx_OD, GEnx_OD_true, N1cCEOD = pickle.load(open("CEOD_GEnx/CEOD_set_Valid.p", "rb"))
_, All_Reynolds = pickle.load(open("Constants/Reynolds_set_Valid.p", "rb"))

All_Reynolds = np.array([item for sublist in All_Reynolds for item in sublist])
GEnx_OD = np.array([item for sublist in GEnx_OD for item in sublist])
GEnx_OD_true = np.array([item for sublist in GEnx_OD_true for item in sublist])
Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19 = All_Reynolds.T




# for j in range(len(All_Reynolds[0])):  # 0: takeoff 1:climb 2:cruise
#     print(j)
#     plt.scatter(GEnx_OD[:, 0], All_Reynolds[:, j], label=output_list[j + 6])
#     plt.xlabel('Corrected Fan Speed [%]')
#     plt.ylabel('Re')
#     plt.legend()
#     plt.show()


# np.random.seed(0)
# batch_size = 45
# centers = [[1, 1], [-1, -1], [1, -1]]
# n_clusters = len(centers)
#
# X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
# print(X.shape)
Re25 = np.vstack((Re25, GEnx_OD[:, 0])).T
# print(Re25.shape)

def create_clusters():
    k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
    k_means.fit(Re25)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)

    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    plt.figure()
    print(k_means_cluster_centers)

    for k, col in zip(range(3), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        # plt.plot(Re25[my_members, 0], Re25[my_members, 1], 'w',
        #          markerfacecolor=col, marker='.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)
    plt.title('KMeans')
    plt.grid()
    plt.show()

create_clusters()
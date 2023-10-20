import os
import pickle
import numpy as np

import sys
from matplotlib import pyplot as plt

from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# file_name = "CEOD_set_valid.p"


def group_CEOD():
    CEOD_pickles = os.listdir('CEOD_GEnx\same_engine_flights')
    # CEOD_pickles.remove('csv files')
    # CEOD_pickles.remove('CEOD_set_Valid.p')

    take_off_GEnx_OD_grouped = np.empty((0, 5))
    climb_GEnx_GEnx_OD_grouped = np.empty((0, 5))
    cruise_GEnx_OD_grouped = np.empty((0, 5))

    take_off_GEnx_OD_true_grouped = np.empty((0, 6))
    climb_GEnx_GEnx_OD_true_grouped = np.empty((0, 6))
    cruise_GEnx_OD_true_grouped = np.empty((0, 6))

    # GEnx_OD_true_grouped = np.array([])

    training_limit = int(np.round(len(CEOD_pickles) / 2, 1))
    print('number of files', len(CEOD_pickles))
    print('number of trained flights', training_limit)
    for file in CEOD_pickles[:training_limit]:
        GEnx_OD, GEnx_OD_true, N, alt_time = pickle.load(open("CEOD_GEnx/same_engine_flights/" + file, "rb"))

        take_off_GEnx_OD, climb_GEnx_OD, cruise_GEnx_OD = GEnx_OD

        take_off_GEnx_OD_grouped = np.concatenate((take_off_GEnx_OD_grouped, take_off_GEnx_OD), axis=0)
        climb_GEnx_GEnx_OD_grouped = np.concatenate((climb_GEnx_GEnx_OD_grouped, climb_GEnx_OD), axis=0)
        cruise_GEnx_OD_grouped = np.concatenate((cruise_GEnx_OD_grouped, cruise_GEnx_OD), axis=0)

        take_off_GEnx_OD_true, climb_GEnx_OD_true, cruise_GEnx_OD_true = GEnx_OD_true

        take_off_GEnx_OD_true_grouped = np.concatenate((take_off_GEnx_OD_true_grouped, take_off_GEnx_OD_true), axis=0)
        climb_GEnx_GEnx_OD_true_grouped = np.concatenate((climb_GEnx_GEnx_OD_true_grouped, climb_GEnx_OD_true), axis=0)
        cruise_GEnx_OD_true_grouped = np.concatenate((cruise_GEnx_OD_true_grouped, cruise_GEnx_OD_true), axis=0)

    return (take_off_GEnx_OD_grouped, climb_GEnx_GEnx_OD_grouped, cruise_GEnx_OD_grouped), \
        (take_off_GEnx_OD_true_grouped, climb_GEnx_GEnx_OD_true_grouped, cruise_GEnx_OD_true_grouped)


def create_clusters(dataset, n_clusters):
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    k_means.fit(dataset)
    k_means_cluster_centers = k_means.cluster_centers_
    return k_means_cluster_centers


def filter_outliers(data_array):
    print(data_array.shape)

    for i in range(data_array.shape[1]):
        data = data_array[:, i]
        running = True
        print(f'data size at the beginning {data_array.shape}')

        if len(data) == 0:
            print("data is empty")
            pass
            # filtered_data.append([])
        else:
            while running:
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                condition = (data >= lower_bound) & (data <= upper_bound)
                if False in condition:
                    data = data[condition]
                    data_array = data_array[condition]
                else:
                    running = False
    return data_array


def clustering(save=False):
    sim_input, sim_output = group_CEOD()
    take_off_GEnx_OD_grouped, climb_GEnx_GEnx_OD_grouped, cruise_GEnx_OD_grouped = sim_input
    take_off_GEnx_OD_true_grouped, climb_GEnx_GEnx_OD_true_grouped, cruise_GEnx_OD_true_grouped = sim_output

    stacked_take_off = np.concatenate(((take_off_GEnx_OD_grouped,
                                        take_off_GEnx_OD_true_grouped)),
                                      axis=1)
    stacked_climb = np.concatenate(((climb_GEnx_GEnx_OD_grouped,
                                     climb_GEnx_GEnx_OD_true_grouped)),
                                   axis=1)

    stacked_cruise = np.concatenate(((cruise_GEnx_OD_grouped,
                                      cruise_GEnx_OD_true_grouped)),
                                    axis=1)
    stacked_take_off = filter_outliers(stacked_take_off)
    stacked_climb = filter_outliers(stacked_climb)
    stacked_cruise = filter_outliers(stacked_cruise)

    clusters_take_off_array = create_clusters(stacked_take_off, 10)
    clusters_climb_array = create_clusters(stacked_climb, 5)
    clusters_cruise_array = create_clusters(stacked_cruise, 5)

    def viz(norm_array, clusters_array):
        colors = ['#4EACC5', '#FF9C34', '#4E9A06']
        idx = 2
        plt.scatter(norm_array[:, 0], norm_array[:, idx], color=colors[0], marker='.')

        plt.plot(clusters_array[:, 0], clusters_array[:, idx], 'o', markerfacecolor=colors[1],
                 markeredgecolor='k', markersize=6)
        plt.show()

    viz(stacked_take_off, clusters_take_off_array)
    viz(stacked_climb, clusters_climb_array)
    viz(stacked_cruise, clusters_cruise_array)

    Genx_input_array = np.concatenate((clusters_take_off_array[:, :5],
                                       clusters_climb_array[:, :5],
                                       clusters_cruise_array[:, :5]))

    Genx_true_array = np.concatenate((clusters_take_off_array[:, 5:],
                                      clusters_climb_array[:, 5:],
                                      clusters_cruise_array[:, 5:]))

    print(Genx_input_array.shape)

    if save:
        pickle.dump([Genx_input_array, Genx_true_array], open("Clusters/Clusters_v1.p", "wb"))


if __name__ == '__main__':
    clustering(save=True)

import pickle
import numpy as np
from matplotlib import pyplot as plt


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
def reduce_points(save=False):
    file = "170108-234714-KLM706____-SBGLEHAM-KL_PH-BHA-2-956609-W007FFD.p"
    GEnx_OD, GEnx_OD_true, _, alt_time = pickle.load(open("CEOD_GEnx/CEOD_" + file, "rb"))
    take_off_GEnx_OD, climb_GEnx_OD, cruise_GEnx_OD = GEnx_OD
    take_off_alt_time, climb_alt_time, cruise_alt_time = alt_time

    take_off_GEnx_OD_grouped, climb_GEnx_GEnx_OD_grouped, cruise_GEnx_OD_grouped = GEnx_OD
    take_off_GEnx_OD_true_grouped, climb_GEnx_GEnx_OD_true_grouped, cruise_GEnx_OD_true_grouped = GEnx_OD_true

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

    def sampling(arr, no_of_samples):
        sample_indices = np.linspace(0, len(arr)-1, no_of_samples, dtype=int)
        if len(arr) - 1 not in sample_indices:
            sample_indices = np.insert(np.delete(sample_indices, -1), len(arr) - 1)
        return arr[sample_indices]

    def viz(norm_array, sampled_array):
        colors = ['#4EACC5', '#FF9C34', '#4E9A06']
        idx = 2
        plt.scatter(norm_array[:, 0], norm_array[:, idx], color=colors[0], marker='.')

        plt.plot(sampled_array[:, 0], sampled_array[:, idx], 'o', markerfacecolor=colors[1],
                 markeredgecolor='k', markersize=6)
        plt.show()

    sampled_take_off = sampling(stacked_take_off, 3)
    sampled_climb = sampling(stacked_climb, 5)
    sampled_cruise = sampling(stacked_cruise, 5)

    viz(stacked_take_off, sampled_take_off)
    viz(stacked_climb, sampled_climb)
    viz(stacked_cruise, sampled_cruise)

    Genx_input_array = np.concatenate((sampled_take_off[:, :5],
                                       sampled_climb[:, :5],
                                       sampled_cruise[:, :5]))

    Genx_true_array = np.concatenate((sampled_take_off[:, 5:],
                                      sampled_climb[:, 5:],
                                      sampled_cruise[:, 5:]))


    if save:
        pickle.dump([Genx_input_array, Genx_true_array], open("Clusters/CEOD_one_flight_sampled_no_Reynolds.p", "wb"))

    # plt.scatter(climb_GEnx_OD[:, 0], climb_GEnx_OD[:, 2])
    # plt.scatter(stacked_climb[:, 0], stacked_climb[:, 2], c='b')
    # plt.show()


    # plt.scatter(take_off_alt_time[:, 1], take_off_alt_time[:, 0])
    # plt.show()
    # plt.scatter(climb_alt_time[:, 1], climb_alt_time[:, 0])
    # plt.show()
    # plt.scatter(cruise_alt_time[:, 1], cruise_alt_time[:, 0])
    # plt.show()


reduce_points(True)
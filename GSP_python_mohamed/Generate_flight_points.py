import pickle
import numpy as np
from matplotlib import pyplot as plt


# def filter_outliers(data_array):
#     for i in range(data_array.shape[1]):
#         data = data_array[:, i]
#         running = True
#         print(f'data size at the beginning {data_array.shape}')
#
#         if len(data) == 0:
#             print("data is empty")
#             pass
#             # filtered_data.append([])
#         else:
#             while running:
#                 q1 = np.percentile(data, 25)
#                 q3 = np.percentile(data, 75)
#                 iqr = q3 - q1
#                 lower_bound = q1 - 1.5 * iqr
#                 upper_bound = q3 + 1.5 * iqr
#                 condition = (data >= lower_bound) & (data <= upper_bound)
#                 if False in condition:
#                     data = data[condition]
#                     data_array = data_array[condition]
#                 else:
#                     running = False
#     return data_array


def reduce_points(save=False):
    file = "Reynolds_CEOD_data_mohamed_2019_feb_1-9_2.p"
    GEnx_OD, GEnx_OD_true, N1c, alt_time, All_Reynolds = pickle.load(open("Reynolds_pickle/"
                                                                          + file,
                                                                          "rb"))

    take_off_GEnx_OD, climb_GEnx_OD, cruise_GEnx_OD = GEnx_OD
    take_off_GEnx_OD_true, climb_GEnx_OD_true, cruise_GEnx_OD_true = GEnx_OD_true
    take_off_alt_time, climb_alt_time, cruise_alt_time = alt_time
    take_off_All_Reynolds, climb_All_Reynolds, cruise_All_Reynolds = All_Reynolds
    #
    # print(take_off_GEnx_OD.shape)
    # print(climb_GEnx_OD.shape)
    # print(cruise_GEnx_OD.shape)

    stacked_take_off = np.concatenate((
        take_off_GEnx_OD,
        take_off_GEnx_OD_true,
        take_off_alt_time,
        take_off_All_Reynolds),
        axis=1)

    stacked_climb = np.concatenate((
        climb_GEnx_OD,
        climb_GEnx_OD_true,
        climb_alt_time,
        climb_All_Reynolds),
        axis=1)

    stacked_cruise = np.concatenate((
        cruise_GEnx_OD,
        cruise_GEnx_OD_true,
        cruise_alt_time,
        cruise_All_Reynolds),
        axis=1)



    def get_sampling_indices(arr, no_of_samples):
        min_val = min(arr)
        max_val = max(arr)

        # Calculate the spacing between min and max for 5 points
        spacing = (max_val - min_val) / (no_of_samples - 1)

        # Initialize a list to store the sampled points
        sample_indices = []

        # Sample 5 points evenly spaced between min and max
        for i in range(no_of_samples):
            target_value = min_val + i * spacing
            closest_value = min(arr, key=lambda x: abs(x - target_value))
            sample_indices.append(np.where(arr == closest_value)[0][0])
        # sample_indices = np.linspace(0, len(arr) - 1, no_of_samples, dtype=int)
        # if len(arr) - 1 not in sample_indices:
        #     sample_indices = np.insert(np.delete(sample_indices, -1), len(arr) - 1)
        return sample_indices

    def viz(norm_array, sampled_array, phase):
        colors = ['#4EACC5', '#FF9C34', '#4E9A06']
        idx = 4
        plt.title(idx)
        plt.scatter(norm_array[:, 12], norm_array[:, 0], color=colors[0], marker='.')

        plt.plot(sampled_array[:, 12], sampled_array[:, 0], 'o', markerfacecolor=colors[1],
                 markeredgecolor='k', markersize=6)
        plt.show()

    no_of_samples = 5
    sampled_take_off = stacked_take_off[get_sampling_indices(stacked_take_off[:, 0], no_of_samples)]
    sampled_climb = stacked_climb[get_sampling_indices(stacked_climb[:, 11], no_of_samples)]
    sampled_cruise = stacked_cruise[get_sampling_indices(stacked_cruise[:, 12], no_of_samples)]

    viz(stacked_take_off, sampled_take_off, "take off")
    # viz(stacked_climb, sampled_climb, "climb")
    # viz(stacked_cruise, sampled_cruise, "cruise")

    Genx_input_array = np.concatenate((sampled_take_off[:, :5],
                                       sampled_climb[:, :5],
                                       sampled_cruise[:, :5]))

    Genx_true_array = np.concatenate((sampled_take_off[:, 5:11],
                                      sampled_climb[:, 5:11],
                                      sampled_cruise[:, 5:11]))

    Alt_time_array = np.concatenate((sampled_take_off[:, 11:13],
                                     sampled_climb[:, 11:13],
                                     sampled_cruise[:, 11:13]))

    All_Reynolds_array = np.concatenate((sampled_take_off[:, 13:],
                                         sampled_climb[:, 13:],
                                         sampled_cruise[:, 13:]))
    # print(Genx_input_array.shape)
    # print(Genx_true_array.shape)
    # print(Alt_time_array.shape)
    # print(All_Reynolds_array.shape)

    new_name = file.strip('.p') + '_sampled'
    if save:
        pickle.dump([Genx_input_array, Genx_true_array, Alt_time_array, All_Reynolds_array],
                    open(f"Sampled flights/{new_name}.p", "wb"))

    # plt.scatter(climb_GEnx_OD[:, 0], climb_GEnx_OD[:, 2])
    # plt.scatter(stacked_climb[:, 0], stacked_climb[:, 2], c='b')
    # plt.show()

    # plt.scatter(take_off_alt_time[:, 1], take_off_alt_time[:, 0])
    # plt.show()
    # plt.scatter(climb_alt_time[:, 1], climb_alt_time[:, 0])
    # plt.show()
    # plt.scatter(cruise_alt_time[:, 1], cruise_alt_time[:, 0])
    # plt.show()


reduce_points(save=True)

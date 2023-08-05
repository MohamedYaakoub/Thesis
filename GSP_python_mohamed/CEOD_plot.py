import pickle
import os
import numpy as np
from matplotlib import pyplot as plt


# for file in os.listdir("CEOD_GEnx/same_engine_flights/"):
#     print(file)
#     GEnx_OD, GEnx_OD_true, _, time_alt = pickle.load(open("CEOD_GEnx/same_engine_flights/" + file, "rb"))
#
#     print(time_alt[1].shape)
#
#     for i in range(3):
#         plt.scatter(time_alt[i][:, 1], time_alt[i][:, 0], c='b')

# GEnx_OD, GEnx_OD_true, _, time_alt = pickle.load(open("CEOD_GEnx/same_engine_flights/"
#                                                       "CEOD_data_mohamed_2019_feb_1-9_0.p", "rb"))


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


def viz_Re():
    filter_val = 10000
    GEnx_OD, GEnx_OD_true, N1c, time_alt, All_Reynolds = pickle.load(open("Reynolds_pickle/"
                                                                        "Reynolds_CEOD_data_mohamed_2019_feb_1-9_1.p",
                                                                        "rb"))

    # names = ["Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19"]
    # for i, name in enumerate(names):
    #     plt.title(f"Reynolds number at station {name[2:]}")
    #     plt.ylabel("RE [-]")
    #     plt.xlabel("Offset")
    #
    #     for idx in range(3):
    #         # Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19 = All_Reynolds[idx].T
    #         offset_filter = time_alt[idx][:, 1] < filter_val
    #         Re = All_Reynolds[idx][:, i]
    #
    #         plt.scatter(time_alt[idx][:, 1][offset_filter], Re[offset_filter], alpha=0.5)
    #     plt.show()

    for idx in range(3):
        # Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19 = All_Reynolds[idx].T
        offset_filter = time_alt[idx][:, 1] < filter_val

        plt.scatter(time_alt[idx][:, 1][offset_filter], time_alt[idx][:, 0][offset_filter], alpha=0.5)
    plt.show()


    # inputs_list = ["N1", "P0", "T0", "Mach", "HP"]
    # for i, name in enumerate(inputs_list):
    #     plt.title(name)
    #     plt.ylabel(name)
    #     plt.xlabel("Offset")
    #     for idx in range(3):
    #         offset_filter = time_alt[idx][:, 1] < filter_val
    #         plt.scatter(time_alt[idx][:, 1][offset_filter], GEnx_OD[idx][:, i][offset_filter], alpha=0.5)
    #     plt.show()

    for idx in range(3):
        offset_filter = time_alt[idx][:, 1] < filter_val
        plt.title("N1C")
        plt.scatter(time_alt[idx][:, 1][offset_filter], N1c[idx][offset_filter], alpha=0.5)
    plt.show()

    # g = 1.4
    # offset_filter = time_alt[1][:, 1] < filter_val
    # isentr = 1 + (g - 1) * 0.5 * GEnx_OD[1][:, 3][offset_filter] ** 2
    # rate = isentr ** (g / (g - 1))
    #
    # plt.title("compressibility correction factor")
    # plt.scatter(time_alt[1][:, 1][offset_filter], rate, alpha=0.5)
    # plt.show()




if __name__ == '__main__':
    viz_Re()
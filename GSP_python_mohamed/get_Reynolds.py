import pickle
import numpy as np
import timeit
import sys
from GSP_helper import cleanup, runGsp
import subprocess
# from Reynolds_effect_functions import scale_maps_reynolds
from matplotlib import pyplot as plt
from WriteCMapGSP import read_mapC, write_mapC
from WriteTMapGSP import read_mapT, write_mapT
from map_functions import reset_maps

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesignGEnx Valid_Shivan.mxl"  # "GEnx-1B_V3_test2.mxl"  #


inputs_list = ["N1", "P0", "T0", "Mach", "HP"]
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2",
               "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19", "Re_6", "Re_9"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))

reset_maps()
from my_modified_functions import gspdll

def get_Reynolds(ceod_file):
    GEnx_OD, GEnx_OD_true, N, alt_time = pickle.load(open("CEOD_GEnx/same_engine_flights/" + ceod_file, "rb"))
    def simulate(inputDat):
        gspdll.InitializeModel()
        y_sim = np.array(runGsp(gspdll, inputDat, output_list))
        Reynolds = y_sim[:, 6:]
        return Reynolds

    All_Reynolds = []
    for inputDat in GEnx_OD:
        print('Loop initiated')
        print(inputDat.shape)

        Reynolds = simulate(inputDat)
        All_Reynolds.append(Reynolds)

    pickle.dump([GEnx_OD, GEnx_OD_true, N, alt_time, All_Reynolds], open("Reynolds_pickle/Reynolds_" + file_name, "wb"))

def filter_outliers(data_array):
    filtered_data = []
    data_before_filtering = 0

    for data in data_array:
        running = True
        print(f'data size at the beginning {len(data)}')
        if len(data) == 0:
            filtered_data.append([])
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
                else:
                    running = False
            filtered_data.append(data)

    return filtered_data


if __name__ == '__main__':

    # file_name = "CEOD_one_flight_sampled_no_Reynolds.p"
    file_name = "CEOD_data_mohamed_2019_feb_1-9_1.p"


    # file_name = "CEOD_200408-203904-KLM168____-KATLEHAM-KL_PH-BHA-2-956609-W010FFD.P"
    # file_name = "CEOD_160724-193429-KLM891____-EHAMZUUU-KL_PH-BHA-2-956609-W007FFD.p"
    get_Reynolds(file_name)

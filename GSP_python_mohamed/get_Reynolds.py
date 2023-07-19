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
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2", "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))

reset_maps()
from my_modified_functions import gspdll

def get_Reynolds(ceod_file):
    GEnx_OD, GEnx_OD_true = pickle.load(open("CEOD_GEnx/CEOD_" + ceod_file, "rb"))
    def simulate(inputDat):
        gspdll.InitializeModel()
        y_sim = np.array(runGsp(gspdll, inputDat, output_list))
        Reynolds = y_sim[:, 6:]
        return Reynolds

    # All_Reynolds = []
    # for inputDat in GEnx_OD:
    #     print('Loop initiated')
    #     print(inputDat.shape)

    print(GEnx_OD.shape)
    All_Reynolds = simulate(GEnx_OD)
        # All_Reynolds.append(Reynolds)

    pickle.dump([GEnx_OD, GEnx_OD_true, All_Reynolds], open("Clusters/Reynolds_" + file_name, "wb"))

if __name__ == '__main__':
    # file_name = "CEOD_one_flight_sampled_no_Reynolds.p"
    file_name = "170108-234714-KLM706____-SBGLEHAM-KL_PH-BHA-2-956609-W007FFD.p"
    # file_name = "CEOD_200408-203904-KLM168____-KATLEHAM-KL_PH-BHA-2-956609-W010FFD.P"
    # file_name = "CEOD_160724-193429-KLM891____-EHAMZUUU-KL_PH-BHA-2-956609-W007FFD.p"
    get_Reynolds(file_name)

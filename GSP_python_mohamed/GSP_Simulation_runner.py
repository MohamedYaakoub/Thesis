import pickle
import numpy as np
import timeit
import sys
from GSP_helper import cleanup, runGsp
from matplotlib import pyplot as plt
from WriteCMapGSP import read_mapC, write_mapC
from WriteTMapGSP import read_mapT, write_mapT

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesignGEnx Valid_Shivan.mxl"  # "GEnx-1B_V3_test2.mxl"  #

GEnx_OD, GEnx_OD_true, N1cCEOD = pickle.load(open("CEOD_input.p", "rb"))

# print(GEnx_OD[1].shape)  # inputDat #this is used for y_sim
# print(GEnx_OD_true[2].shape)  # TrueVal of CEOD ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2"]

inputs_list = ["N1", "P0", "T0", "Mach", "HP"]
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2", "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))

GEnx_OD = np.array([item for sublist in GEnx_OD for item in sublist])
GEnx_OD_true = np.array([item for sublist in GEnx_OD_true for item in sublist])

modified_input = np.copy(GEnx_OD)
new_P0 = np.ones((GEnx_OD.shape[0]))
new_T0 = np.ones((GEnx_OD.shape[0])) * 285

modified_input[:, 1] = new_P0
modified_input[:, 2] = new_T0
from my_modified_functions import gspdll

def run_simulation(input):
    gspdll.InitializeModel()
    y_sim_iter = np.array(runGsp(gspdll, input, output_list))
    y_sim = y_sim_iter[:, :6]  # ignore effs for now
    Reynolds_vals = y_sim_iter[:, 6:]
    return Reynolds_vals

Reynolds_vals_CEOD = run_simulation(GEnx_OD)
Reynolds_vals_modified_input = run_simulation(modified_input)

for i in range(len(Reynolds_vals_CEOD[0])):  # 0: takeoff 1:climb 2:cruise
    plt.scatter(GEnx_OD[:, 0], Reynolds_vals_CEOD[:, i], label=output_list[i + 6])
    plt.scatter(GEnx_OD[:, 0], Reynolds_vals_modified_input[:, i], label=output_list[i + 6] + " modified sea level")
    plt.xlabel('Corrected Fan Speed [%]')
    plt.ylabel('Re')
    plt.legend()
    plt.show()

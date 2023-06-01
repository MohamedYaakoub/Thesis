"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 8/19/2022 4:37 PM  
"""
import pickle
import numpy as np
import timeit
import sys
from GSP_helper import cleanup, runGsp
import subprocess

sys.path.insert(1, "C:/Users/mohsy/University/KLM/Thesis/My thesis/Parallel GSP/Shared "
                   "Folder/GSP")
from matplotlib import pyplot as plt

GEnx_OD, GEnx_OD_true, N1cCEOD = pickle.load(open("CEOD_input.p", "rb"))  # deal with large data set (memory problem

# %%
Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesignGEnx Valid.mxl"  #"GEnx-1B_V3_test2.mxl"  #
# note Valid is the latest model-6
# Valid 2 is uncalibrated
# Valid C2 is another calibrated model, developed not using the initial guess

# note: these should be identical to the parameters defined in the API module of GSP

inputs_list = ["N1", "P0", "T0", "Mach", "HP", "Cv"]
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2", "FanBp", "FanC", "HPC", "HPT", "LPT"] #, "PRfanc", "PRHPC"]

# dump the following to transfer them to the objective function file
pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))
# import the objective function
from OD_function import gspdll


# %%
def compute_error(inputDat, trueVal):
    y_sim        = np.array(runGsp(gspdll, inputDat, output_list))
    efficiencies = y_sim[:, 6:]
    y_sim     = y_sim[:, :6]
    change    = (trueVal - y_sim)/(trueVal+0.000001)
    meanE     = 100*np.sqrt(np.mean(change**2, axis=0))
    return meanE, change*100, np.mean(efficiencies, axis=0)

meanL = []
stdL  = []
stdLR = []
EtaL  = []

Etas        = output_list[6:]
paramE      = output_list[:6]

for i, j, k in zip(GEnx_OD, GEnx_OD_true, N1cCEOD):
    trueVal  = j
    inputDat = i
    N1cCEODi = k
    mean, change, Eta  = compute_error(inputDat, trueVal)
    meanL.append(list(mean))
    stdL.append(list(np.std(change, axis=0)))
    EtaL.append(list(Eta*100))
    stdLRi = []
    for parameterE in change.T:
        coef = np.polyfit(inputDat[:, 0], parameterE, 1)

        func = np.poly1d(coef)
        yf = func(inputDat[:, 0])
        err = yf - parameterE
        stdLRi.append(sum(np.abs(err))/len(err))
    stdLR.append(stdLRi)


    # %%
    cmap  = plt.get_cmap('tab20')
    clist = cmap(np.linspace(0, 1, len(paramE)))
    plt.figure()
    for i in range(len(paramE)):
        plt.scatter(inputDat[:,0], change[:, i], label=paramE[i])
    plt.xlabel('Corrected Fan Speed [%]')
    plt.ylabel('Error (CEOD - GSP) [%]')
    plt.legend(loc='lower center')
    # plt.ylim(-10, 6)
    plt.show()

print("Part 1 done")
# %%
from OD_sens import barC

# barC([list(meanL[2])], ['Take-off', 'Climb', 'Cruise'][2], output_list)
barC(meanL, ['Take-off', 'Climb', 'Cruise'], paramE, "Error [%]")

# barC(stdLR, ['Take-off', 'Climb', 'Cruise'], paramE, " MAE [%]")

#%% plotting etas

# etaL2 = np.array(EtaL) - [[88.745256078151, 87.46169221301324, 86.9082962135546, 89.84537895667212, 89.85943110141969, 201.1306368506157, 2048.0910016230678], [87.83020211604273, 83.8319449261757, 85.96903152379595, 89.08484474007153, 88.14901908424379, 235.09091590080695, 2134.939041420237], [87.32206375171961, 86.58636471898677, 85.20719093929084, 88.41644018617744, 86.87372810921822, 201.1593008132521, 2091.5627475029646]]
# etaL2 = [l.tolist() for l in etaL2]
#
# barC(etaL2, ['Take-off', 'Climb', 'Cruise'], ["EtaFanBp", "EtaFanC", "EtaHPC", "EtaHPT", "EtaLPT", "PRfanc", "PRHPC"],
#      "Performance change [absolute]")
#
barC(EtaL, ['Take-off', 'Climb', 'Cruise'], Etas, "Efficiency [%]")


# %%
cleanup(gspdll)

#%%
# import dtale
# import matplotlib
# from matplotlib import pyplot as plt
# # matplotlib.use('module://backend_interagg')

#%%
# numpy.std(array here)
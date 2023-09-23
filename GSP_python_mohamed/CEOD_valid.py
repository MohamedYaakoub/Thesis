"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 8/19/2022 4:37 PM  
"""
import pickle
import numpy as np
from GSP_helper import cleanup, runGsp
from map_functions import reset_maps
from matplotlib import pyplot as plt


# file_name = "CEOD_set_Valid.P"
# GEnx_OD, GEnx_OD_true, N1cCEOD = pickle.load(open("CEOD_GEnx/" + file_name, "rb"))

file_name = "CEOD_data_mohamed_2019_feb_1-9_1.p"
GEnx_OD, GEnx_OD_true, N1cCEOD, alt_time = pickle.load(open("CEOD_GEnx/same_engine_flights/" + file_name, "rb"))
# # %%
Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesignGEnx Valid_Shivan.mxl"  # "GEnx-1B_V3_test2.mxl"  #


# note Valid is the latest model-6
# Valid 2 is uncalibrated
# Valid C2 is another calibrated model, developed not using the initial guess

# note: these should be identical to the parameters defined in the API module of GSP

inputs_list = ["N1", "P0", "T0", "Mach", "HP"]
# output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2", "Re_4"]  # , "PRfanc", "PRHPC"]
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2",
               "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19", "Re6", "Re9"]

# dump the following to transfer them to the objective function file
pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))
# import the objective function

reset_maps()
# from OD_function import gspdll
from my_modified_functions import gspdll


# %%
def compute_error(inputDat, trueVal):
    y_sim = np.array(runGsp(gspdll, inputDat, output_list))
    efficiencies = y_sim[:, 6:]
    Reynolds = y_sim[:, 6:]
    y_sim = y_sim[:, :6]
    change = (trueVal - y_sim) / (trueVal + 0.000001)
    meanE = 100 * np.sqrt(np.mean(change ** 2, axis=0))
    return meanE, change * 100, np.mean(efficiencies, axis=0), Reynolds


meanL = []
stdL = []
stdLR = []
EtaL = []
# Re_4 = []

def run_validation():
    Etas = output_list[6:]
    paramE = output_list[:6]

    All_Reynolds = []
    All_change = []
    for i, j, k in zip(GEnx_OD, GEnx_OD_true, N1cCEOD):
        print('Loop initiated')

        inputDat = i
        trueVal = j
        N1cCEODi = k
        print(inputDat.shape)
        gspdll.InitializeModel()
        mean, change, Eta, Reynolds = compute_error(inputDat, trueVal)
        meanL.append(list(mean))
        stdL.append(list(np.std(change, axis=0)))
        EtaL.append(list(Eta * 100))
        All_Reynolds.append(Reynolds)
        All_change.append(change/100)
        stdLRi = []
        # for parameterE in change.T:
        #     coef = np.polyfit(inputDat[:, 0], parameterE, 1)
        #
        #     func = np.poly1d(coef)
        #     yf = func(inputDat[:, 0])
        #     err = yf - parameterE
        #     stdLRi.append(sum(np.abs(err)) / len(err))
        # stdLR.append(stdLRi)

        # %%
        cmap = plt.get_cmap('tab20')
        clist = cmap(np.linspace(0, 1, len(paramE)))
        plt.figure()
        for i in range(len(paramE)):
            plt.scatter(inputDat[:, 0], change[:, i], label=paramE[i])
        plt.xlabel('Corrected Fan Speed [%]')
        plt.ylabel('Error (CEOD - GSP) [%]')
        plt.legend(loc='lower center')
        # plt.ylim(-10, 6)
        plt.show()

        #
        #
        # for iter, Re in enumerate(Reynolds[0]):
        #     # print(len(Reynolds[0]))
        #     plt.scatter(inputDat[:, 0], Reynolds[:, iter], label=output_list[iter+6])
        #     plt.xlabel('Corrected Fan Speed [%]')
        #     plt.ylabel('Re')
        #
        # plt.legend()
        # plt.show()

    # barC(EtaL, ['Take-off', 'Climb', 'Cruise'], Etas, "Efficiency [%]")
    All_change = [item for sublist in All_change for item in sublist]
    Rms = np.sqrt(np.mean(np.mean(np.array(All_change) ** 2, axis=0)))
    barC(meanL, ['Take-off', 'Climb', 'Cruise'], paramE, "Error [%]",
         f'{file_name.strip("CEOD_").strip(".p")} \n \n RMSE: {str(round(Rms, 6))}')
    print(Rms, "rms")

    print("Part 1 done")


# %%
# from OD_sens import barC
def barC(outputval, selected_k, params_out, y_name, title):
    plt.rcParams['figure.dpi'] = 500
    outp_length = len(outputval[0]) if isinstance(outputval[0], list) else len(outputval)
    len_k = len(selected_k) if len(selected_k) == 3 else 1
    w = 6  # in inch the width of the figure was 6
    h = 3  # height in inch was 3
    r = np.arange(outp_length)
    r = r * w / outp_length
    width = 0.18  # the bar with was0.18
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    colorl = ['coral', 'goldenrod', 'royalblue']

    for i in range(len_k):
        label = selected_k[i] if len(selected_k) == 3 else selected_k
        rec = ax.bar(r + width * i, np.round(outputval[i], 1), color=colorl[i], width=width, edgecolor=colorl[i],
                     label=label,
                     tick_label=outputval[i])
        # ax.bar_label(rec, padding=3)
    ax.yaxis.grid()  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest
    plt.xlabel("Parameters")
    plt.ylabel(y_name)
    # plt.title("Sensitivity ")
    plt.ylim(0, 8.5)
    plt.xticks(r + width * len_k / 2.6, params_out)  # was 2.6
    plt.yticks(np.arange(9))
    plt.legend(loc='upper right')  # lower
    plt.title(title)
    fig.tight_layout()

    plt.margins(y=0.1)
    plt.show()

# for i in range(len(inputs_list[1:])):
#     plt.scatter(GEnx_OD[2][:, 0], GEnx_OD[2][:, i+1])
#     plt.xlabel('Corrected Fan Speed [%]')
#     plt.ylabel(str(inputs_list[i+1]))
#     plt.show()


if __name__ == '__main__':
    run_validation()
    # cleanup(gspdll)


# %%


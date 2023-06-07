from scipy.optimize import differential_evolution, NonlinearConstraint
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

_, All_Reynolds = pickle.load(open("Constants/Reynolds_set_Valid.p", "rb"))

for i in range(3):
    for j in range(len(All_Reynolds[i][0])):  # 0: takeoff 1:climb 2:cruise
        print(j)
        plt.scatter(GEnx_OD[i][:, 0], All_Reynolds[i][:, j], label=output_list[j + 6])
        plt.xlabel('Corrected Fan Speed [%]')
        plt.ylabel('Re')
        plt.legend()
    plt.show()

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
# plt.show()


from my_modified_functions import gspdll
#

def scaling_F(ReDP, ReOD, a, b):
    """
    Scaling function is a second degree polynomial
    :param ReDP: design spool speed
    :param ReOD: off-design spool speed
    :return: function value
    """
    return np.array(1 + a * ((ReOD - ReDP) / ReDP) + b * ((ReOD - ReDP) / ReDP) ** 2)


def scale_maps_reynolds(typef, ReOD_arr, ReDP, file_name, poly_param):
    X = poly_param

    if typef == 'C':
        MdotC, EtaC, PRC, surge_mC, surge_pC, NC = pickle.load(open(file_name + "pick.p", "rb"))
        fpr = scaling_F(ReDP, ReOD_arr, X[0], X[1])
        fm  = scaling_F(ReDP, ReOD_arr, X[2], X[3])
        fe  = scaling_F(ReDP, ReOD_arr, X[4], X[5])

        surgeN = np.unique(NC)

        fsurge_pr = scaling_F(ReDP, surgeN, X[0], X[1])
        fsurge_m  = scaling_F(ReDP, surgeN, X[2], X[3])
        fsurge_m  = np.insert(fsurge_m, 0, 1)
        fsurge_pr = np.insert(fsurge_pr, 0, 1)

        write_mapC(file_name, file_name, np.clip(MdotC * fm, 0.05, 2000), np.clip(EtaC * fe, 0.10101, 0.99),
                   np.clip(PRC * fpr, 0.05, 100), np.clip(surge_mC * fsurge_m, 0.05, 2000),
                   np.clip(surge_pC * fsurge_pr, 0.05, 100), NC)
    else:
        PRmin, PRmax, MdotT, EtaT, NPrT, NT, BT = pickle.load(open(file_name + "pick.p", "rb"))
        # fpr = scaling_F(Ndp / 100, NPrT[1:], X[0], X[1])
        # fm  = scaling_F(Ndp / 100, NT, X[2], X[3])
        fe  = scaling_F(ReDP, ReOD_arr, X[0], X[1])
        # fpr = np.insert(fpr, 0, 1)

        write_mapT(file_name, file_name, np.clip(PRmin * 1, 0, 100), np.clip(PRmax * 1, 0, 100), np.clip(MdotT * 1,
                                                                 0.05, 2000), np.clip(EtaT * fe, 0.10101, 0.99), NT, BT)

def objFOD(X):
    gspdll.InitializeModel()

    scale_maps_reynolds("C", Re25, Re25[0], "1_LPC_core", X[:6])
    scale_maps_reynolds("C", Re19, Re19[0], "2_LPC_bypass", X[6:12])
    scale_maps_reynolds("C", Re3, Re3[0], "3_HPC", X[12:18])
    scale_maps_reynolds("T", Re49, Re49[0], "4_HPT", X[18:20])
    scale_maps_reynolds("T", Re5, Re5[0], "5_LPT", X[20:22])

    inputDat = GEnx_OD
    y_true = GEnx_OD_true

    y_sim = np.array(runGsp(gspdll, inputDat, output_list))
    y_sim = y_sim[:, :6]  # ignore effs for now

    weights = np.ones(6)
    Rms = np.sqrt(np.mean(np.mean(((y_true - y_sim) / (y_true + 0.000001)) ** 2, axis=0) * weights))
    return Rms


# objFOD(12)




# MdotC, EtaC, PRC, surge_mC, surge_pC, NC = pickle.load(open(
#     "C:/Users/mohsy/University/KLM/Thesis/My thesis/Code/GSP_python_shivan/1_LPC_corepick.p", "rb"))
#
# print(np.unique(NC))
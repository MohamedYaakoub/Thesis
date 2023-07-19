from scipy.optimize import differential_evolution, NonlinearConstraint
import pickle
import numpy as np
import timeit
import sys
import subprocess
from GSP_helper import cleanup, runGsp
from matplotlib import pyplot as plt
from WriteCMapGSP import read_mapC, write_mapC
from WriteTMapGSP import read_mapT, write_mapT

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesignGEnx Valid_Shivan.mxl"  # "GEnx-1B_V3_test2.mxl"  #
#
# GEnx_OD, GEnx_OD_true, N1cCEOD = pickle.load(open("CEOD_set_Valid.p", "rb"))

# _, All_Reynolds = pickle.load(open("Constants/Reynolds_set_Valid.p", "rb"))
#
# All_Reynolds = np.array([item for sublist in All_Reynolds for item in sublist])
# GEnx_OD = np.array([item for sublist in GEnx_OD for item in sublist])
# GEnx_OD_true = np.array([item for sublist in GEnx_OD_true for item in sublist])

GEnx_OD, GEnx_OD_true, All_Reynolds = pickle.load(open("Clusters/Reynolds_one_flight_sampled.p", "rb"))

Re25_DP = All_Reynolds[0, 1]
Re3_DP = All_Reynolds[0, 2]
Re49_DP = All_Reynolds[0, 4]
Re5_DP = All_Reynolds[0, 5]
Re19_DP = All_Reynolds[0, 7]

GEnx_OD = GEnx_OD.astype(np.float32)[8:, :]
GEnx_OD_true = GEnx_OD_true.astype(np.float32)[8:, :]
All_Reynolds = All_Reynolds.astype(np.float32)[8:, :]

inputs_list = ["N1", "P0", "T0", "Mach", "HP"]
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2", "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))


# for i in range(3):
#     for j in range(len(All_Reynolds[i][0])):  # 0: takeoff 1:climb 2:cruise
#         print(j)
#         plt.scatter(GEnx_OD[i][:, 0], All_Reynolds[i][:, j], label=output_list[j + 6])
#         plt.xlabel('Corrected Fan Speed [%]')
#         plt.ylabel('Re')
#         plt.legend()
#     plt.show()


Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19 = All_Reynolds.T
# print(Re2.shape)
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


def scale_maps_reynolds(typef, ReDP, ReOD, file_name, poly_param):
    X = poly_param

    if typef == 'C':
        MdotC, EtaC, PRC, surge_mC, surge_pC, NC = pickle.load(open("Constants/" + file_name + "pick.p", "rb"))
        fpr = scaling_F(ReDP, ReOD, X[0], X[1])
        fm  = scaling_F(ReDP, ReOD, X[2], X[3])
        fe  = scaling_F(ReDP, ReOD, X[4], X[5])

        surgeN = np.unique(NC)

        fsurge_pr = scaling_F(ReDP, surgeN, X[0], X[1])
        fsurge_m  = scaling_F(ReDP, surgeN, X[2], X[3])
        fsurge_m  = np.insert(fsurge_m, 0, 1)
        fsurge_pr = np.insert(fsurge_pr, 0, 1)
        # print('MdotC:', MdotC * fm)
        # print('EtaC:', EtaC * fe)
        # print('PRC:', PRC * fpr)

        write_mapC(file_name, file_name, np.clip(MdotC * fm, 0.05, 2000), np.clip(EtaC * fe, 0.10101, 0.99),
                   np.clip(PRC * fpr, 0.05, 100), np.clip(surge_mC * fsurge_m, 0.05, 2000),
                   np.clip(surge_pC * fsurge_pr, 0.05, 100), NC)
    else:
        PRmin, PRmax, MdotT, EtaT, NPrT, NT, BT = pickle.load(open("Constants/" + file_name + "pick.p", "rb"))
        # fpr = scaling_F(Ndp / 100, NPrT[1:], X[0], X[1])
        # fm  = scaling_F(Ndp / 100, NT, X[2], X[3])
        fe  = scaling_F(ReDP, ReOD, X[0], X[1])
        # fpr = np.insert(fpr, 0, 1)

        write_mapT(file_name, file_name, np.clip(PRmin * 1, 0, 100), np.clip(PRmax * 1, 0, 100), np.clip(MdotT * 1,
                                                                 0.05, 2000), np.clip(EtaT * fe, 0.10101, 0.99), NT, BT)


def objFOD(X):
    y_sim = []

    # Re25_DP = 18660698.2305537
    # Re19_DP = 26333708.4295415
    # # Re19_DP = 61973246.778017
    # # Re3_DP = 35472717.4019559
    # Re3_DP = 61973246.778017
    # Re49_DP = 22736943.8967379
    # Re5_DP = 18683167.9100408

    print(X, "evals")

    for i in range(len(GEnx_OD)):

        scale_maps_reynolds("C", Re25_DP, Re25[i], "1_LPC_core", X[:6])
        scale_maps_reynolds("C", Re19_DP, Re19[i], "2_LPC_bypass", X[6:12])
        scale_maps_reynolds("C", Re3_DP, Re3[i], "3_HPC", X[12:18])
        scale_maps_reynolds("T", Re49_DP, Re49[i], "4_HPT", X[18:20])
        scale_maps_reynolds("T", Re5_DP, Re5[i], "5_LPT", X[20:22])

        gspdll.InitializeModel()


        y_sim_iter = np.array(runGsp(gspdll, GEnx_OD[i, :], output_list))
        y_sim_iter = y_sim_iter[:6]  # ignore effs for now
        y_sim.append(y_sim_iter)

    y_sim = np.array(y_sim)
    y_true = GEnx_OD_true
    weights = np.ones(6)
    Rms = np.sqrt(np.mean(np.mean(((y_true - y_sim) / (y_true + 0.000001)) ** 2, axis=0) * weights))
    print(Rms, "rms")
    # print(y_true.shape, y_sim.shape)
    return Rms


# objFOD(12)




# MdotC, EtaC, PRC, surge_mC, surge_pC, NC = pickle.load(open(
#     "C:/Users/mohsy/University/KLM/Thesis/My thesis/Code/GSP_python_shivan/1_LPC_corepick.p", "rb"))
#
# print(np.unique(NC))
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

from my_modified_functions import gspdll

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesignGEnx Nozzles.mxl"  # "GEnx-1B_V3_test2.mxl"  #

GEnx_OD, GEnx_OD_true, Alt_time, All_Reynolds = pickle.load(open("Clusters/Reynolds_CEOD_data_mohamed_2019_feb_1-9_1_sampled.p", "rb"))

Re25_DP = All_Reynolds[0, 1]
Re3_DP = All_Reynolds[0, 2]
Re49_DP = All_Reynolds[0, 4]
Re5_DP = All_Reynolds[0, 5]
Re19_DP = All_Reynolds[0, 7]
Re9_DP = All_Reynolds[0,9]


# GEnx_OD = GEnx_OD.astype(np.float32)[:, :]
# GEnx_OD_true = GEnx_OD_true.astype(np.float32)[:, :]
# All_Reynolds = All_Reynolds.astype(np.float32)[:, :]

inputs_list = ["N1", "P0", "T0", "Mach", "HP", "CX_c", "CV_c", "CX_b", "CV_b"]
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2", "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))



Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19, Re6, Re9 = All_Reynolds.T


def scaling_F(ReDP, ReOD, a, b, c):
    """
    Scaling function is a second degree polynomial
    :param ReDP: design spool speed
    :param ReOD: off-design spool speed
    :return: function value
    """
    return np.array(c + a * ((ReOD - ReDP) / ReDP) + b * ((ReOD - ReDP) / ReDP) ** 2).reshape(-1, 1)


def objFOD(X):
    # y_sim = []

    Cx_ub = 1.05
    Cx_lb = 0.80

    Cv_ub = 1.00
    Cv_lb = 0.95

    CX_c = scaling_F(Re9_DP, Re9, X[0], X[1], 0.9376)
    CX_c = np.clip(CX_c, Cx_lb, Cx_ub)

    CV_c = scaling_F(Re9_DP, Re9, X[2], X[3], 1)
    CV_c = np.clip(CV_c, Cv_lb, Cv_ub)

    CX_d = scaling_F(Re19_DP, Re19, X[4], X[5], 0.93)
    CX_d = np.clip(CX_d, Cx_lb, Cx_ub)

    CV_d = scaling_F(Re19_DP, Re19, X[6], X[7], 1)
    CV_d = np.clip(CV_d, Cv_lb, Cv_ub)

    simulation_input = np.concatenate((GEnx_OD, CX_c, CV_c, CX_d, CV_d), axis=1)

    print(simulation_input.shape)
    print(X, "evals")
    # print(f'Core (Cx, Cv) {CX_c, CV_c}', f'Duct (Cx, Cv) {CX_d, CV_d}')

    gspdll.InitializeModel()
    y_sim_iter = np.array(runGsp(gspdll, simulation_input, output_list))

    y_sim_iter = y_sim_iter[:, :6]  # ignore effs for now

    y_true = GEnx_OD_true
    weights = np.ones(6)
    Rms = np.sqrt(np.mean(np.mean(((y_true - y_sim_iter) / (y_true + 0.000001)) ** 2, axis=0) * weights))
    print(Rms, "rms")
    # print(y_true.shape, y_sim.shape)
    return Rms

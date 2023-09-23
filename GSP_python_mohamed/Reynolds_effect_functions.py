import pickle
import numpy as np
from GSP_helper import cleanup, runGsp

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesignGEnx OD Scaling.mxl"  # "GEnx-1B_V3_test2.mxl"  #

# All_Reynolds = np.array([item for sublist in All_Reynolds for item in sublist])
# GEnx_OD = np.array([item for sublist in GEnx_OD for item in sublist])
# GEnx_OD_true = np.array([item for sublist in GEnx_OD_true for item in sublist])

file_name = "Reynolds_CEOD_data_mohamed_2019_feb_1-9_2_sampled"
GEnx_OD, GEnx_OD_true, _, All_Reynolds = pickle.load(open(f"Sampled flights/{file_name}.p", "rb"))

print('Sampled data shape:', GEnx_OD.shape)
GEnx_OD = GEnx_OD.astype(np.float32)[5:10, :]
GEnx_OD_true = GEnx_OD_true.astype(np.float32)[5:10, :]
All_Reynolds = All_Reynolds.astype(np.float32)[5:10, :]

Re2_DP = All_Reynolds[0, 0]
Re25_DP = All_Reynolds[0, 1]
Re3_DP = All_Reynolds[0, 2]
Re4_DP = All_Reynolds[0, 3]
Re49_DP = All_Reynolds[0, 4]
Re5_DP = All_Reynolds[0, 5]
Re19_DP = All_Reynolds[0, 7]

# GEnx_OD = GEnx_OD[10:, :]
# GEnx_OD_true = GEnx_OD_true[10:, :]
# All_Reynolds = All_Reynolds[10:, :]


OG_list = ["N1", "P0", "T0", "Mach", "HP"]
OD_input_list = ["ODSF_Wc core", "ODSF_PR core", "ODSF_Eta core",  # Fan core
                 "ODSF_Wc duct", "ODSF_PR duct", "ODSF_Eta duct",  # Fan duct
                 "ODSF_Wc", "ODSF_PR", "ODSF_Eta",  # HPC
                 "ODSF_Wc", "ODSF_PR", "ODSF_Eta",  # HPT
                 "ODSF_Wc", "ODSF_PR", "ODSF_Eta"]  # LPT

inputs_list = OG_list + OD_input_list
# print(f'Input list size {len(inputs_list)}')

output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2", "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))

Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19, _, _ = All_Reynolds.T

from my_modified_functions import gspdll


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
        fm = scaling_F(ReDP, ReOD, X[0], X[1])
        # fpr = scaling_F(ReDP, ReOD, X[0], X[1])
        fe = scaling_F(ReDP, ReOD, X[2], X[3])
        return fm, fe
    else:
        fe = scaling_F(ReDP, ReOD, X[0], X[1])
        return fe

# def scale_maps_reynolds(typef, ReDP, ReOD, file_name, poly_param):
#     X = poly_param
#
#     if typef == 'C':
#         fm = scaling_F(ReDP, ReOD, X[2], X[3])
#         # fpr = scaling_F(ReDP, ReOD, X[0], X[1])
#         fe = scaling_F(ReDP, ReOD, X[4], X[5])
#         return fm, fe
#     else:
#         fe = scaling_F(ReDP, ReOD, X[0], X[1])
#         return fe

def objFOD(X):
    y_sim = []

    print(X, "evals")
    gspdll.InitializeModel()

    for i in range(len(GEnx_OD)):
        Fan_c_fm, Fan_c_fe = scale_maps_reynolds("C", Re2_DP, Re2[i], "1_LPC_core", X[:4])
        Fan_d_fm,  Fan_d_fe = scale_maps_reynolds("C", Re2_DP, Re2[i], "2_LPC_bypass", X[4:8])
        HPC_fm, HPC_fe = scale_maps_reynolds("C", Re25_DP, Re25[i], "3_HPC", X[8:12])
        HPT_fe = scale_maps_reynolds("T", Re4_DP, Re4[i], "4_HPT", X[12:14])
        LPT_fe = scale_maps_reynolds("T", Re49_DP, Re49[i], "5_LPT", X[14:16])

        # Fan_c_fm, Fan_c_fe = scale_maps_reynolds("C", Re2_DP, Re2[i], "1_LPC_core", X[:6])
        # Fan_d_fm, Fan_d_fe = scale_maps_reynolds("C", Re2_DP, Re2[i], "2_LPC_bypass", X[6:12])
        # HPC_fm, HPC_fe = scale_maps_reynolds("C", Re25_DP, Re25[i], "3_HPC", X[12:18])
        # HPT_fe = scale_maps_reynolds("T", Re4_DP, Re4[i], "4_HPT", X[18:20])
        # LPT_fe = scale_maps_reynolds("T", Re49_DP, Re49[i], "5_LPT", X[20:22])

        OD_scaling_array = np.array([Fan_c_fm, 1, Fan_c_fe,
                                     Fan_d_fm, 1, Fan_d_fe,
                                     HPC_fm, 1, HPC_fe,
                                     1, 1, HPT_fe,
                                     1, 1, LPT_fe])

        run_gsp_input = np.concatenate((GEnx_OD[i, :], OD_scaling_array), axis=0)
        y_sim_iter = runGsp(gspdll, run_gsp_input, output_list)
        y_sim_iter = y_sim_iter[:6]  # ignore effs for now
        y_sim.append(y_sim_iter)

    y_sim = np.array(y_sim)
    y_true = GEnx_OD_true
    weights = np.ones(6)
    Rms = np.sqrt(np.mean(np.mean(((y_true - y_sim) / (y_true + 0.000001)) ** 2, axis=0) * weights))

    print(Rms, "rms")
    # print(y_true.shape, y_sim.shape)
    return Rms


# objFOD(22 * [0])

import pickle
import numpy as np
from GSP_helper import cleanup, runGsp
from matplotlib import pyplot as plt

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesignGEnx_Valid_All parameters.mxl"  # "GEnx-1B_V3_test2.mxl"  #

file_name = "Reynolds_CEOD_data_mohamed_2019_feb_1-9_2_sampled"
GEnx_OD, _, _, All_Reynolds = pickle.load(open(f"Sampled flights/{file_name}.p", "rb"))

OG_list = ["N1", "P0", "T0", "Mach", "HP"]
OD_input_list = ["ODSF_Wc core", "ODSF_PR core", "ODSF_Eta core",  # Fan core
                 "ODSF_Wc duct", "ODSF_PR duct", "ODSF_Eta duct",  # Fan duct
                 "ODSF_Wc", "ODSF_PR", "ODSF_Eta",  # HPC
                 "ODSF_Wc", "ODSF_PR", "ODSF_Eta",  # HPT
                 "ODSF_Wc", "ODSF_PR", "ODSF_Eta"]  # LPT

inputs_list = OG_list + OD_input_list
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2",
               "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19", "Re_6", "Re_9",
               "OPR", "PRc_Fan", "PRd_Fan", "PR_HPC", "PR_HPT", "PR_LPT", "PR_d", "PR_nozzle", "PR_bypass",
               "ETAc_Fan", "ETAd_Fan", "Eta_HPC", "Eta_HPT", "Eta_LPT",
               "W14", "W25", "W4", "Wc2", "Wcc_fan", "Wcd_Fan", "Wc25", "Wc4", "Wc49",
               "TRc_fan", "TRd_fan", "TR_HPC", "TR_HPT", "TR_LPT", "TR_d", "TR_nozzle", "TR_bypass",
               "FN", "FG_nozzle", "FG_bypass",
               "Tt2", "Tt14", "Tt25", "Tt3", "Tt4", "Tt49", "Tt5",
               "Pt2", "Pt14", "Pt25", "Pt3", "Pt4", "Pt49", "Pt5",
               "Nc_fan", 'Nc_HPC', 'Nc_HPT', 'Nc_LPT',
               "Betac_fan", 'Betad_fan', 'Beta_HPC', 'Beta_HPT', 'Beta_LPT',
               "M25", "M14", 'M3', "M49", "M5"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))

# GEnx_OD = np.array([item for sublist in GEnx_OD for item in sublist])
# GEnx_OD_true = np.array([item for sublist in GEnx_OD_true for item in sublist])


# def scaling_F(ReDP, ReOD, a, b):
#     """
#     Scaling function is a second degree polynomial
#     :param ReDP: design spool speed
#     :param ReOD: off-design spool speed
#     :return: function value
#     """
#     return np.array(1 + a * ((ReOD - ReDP) / ReDP) + b * ((ReOD - ReDP) / ReDP) ** 2)


# def scale_maps_reynolds(typef, ReDP, ReOD, file_name, poly_param):
#     X = poly_param
#
#     if typef == 'C':
#         fm = scaling_F(ReDP, ReOD, X[2], X[3])
#         fpr = scaling_F(ReDP, ReOD, X[0], X[1])
#         fe = scaling_F(ReDP, ReOD, X[4], X[5])
#         return fm, fpr, fe
#     else:
#         fe = scaling_F(ReDP, ReOD, X[0], X[1])
#         return fe
#

GEnx_OD = np.tile(GEnx_OD[14:15, :], [13, 1])

Fan_c_fm, Fan_c_fpr, Fan_c_fe = 1, 1, 1
Fan_d_fm, Fan_d_fpr, Fan_d_fe = 1, 1, 1
HPC_fm, HPC_fpr, HPC_fe = 1, 1, 1
HPT_fm, HPT_fpr, HPT_fe = 1, 1, 1
LPT_fm, LPT_fpr, LPT_fe = 1, 1, 1


def get_input(scaling_type):
    def get_OD_scaling_array(fm, pr, eta):

        OD_scaling_array_Fanc = np.array([Fan_c_fm + fm, Fan_c_fpr + pr, Fan_c_fe + eta,
                                          Fan_d_fm, Fan_d_fpr, Fan_d_fe,
                                          HPC_fm, HPC_fpr, HPC_fe,
                                          HPT_fm, HPT_fpr, HPT_fe,
                                          LPT_fm, LPT_fpr, LPT_fe]).reshape((1, 15))

        OD_scaling_array_Fand = np.array([Fan_c_fm, Fan_c_fpr, Fan_c_fe,
                                          Fan_d_fm + fm, Fan_d_fpr + pr, Fan_d_fe + eta,
                                          HPC_fm, HPC_fpr, HPC_fe,
                                          HPT_fm, HPT_fpr, HPT_fe,
                                          LPT_fm, LPT_fpr, LPT_fe]).reshape((1, 15))

        OD_scaling_array_HPC = np.array([Fan_c_fm, Fan_c_fpr, Fan_c_fe,
                                         Fan_d_fm, Fan_d_fpr, Fan_d_fe,
                                         HPC_fm + fm, HPC_fpr + pr, HPC_fe + eta,
                                         HPT_fm, HPT_fpr, HPT_fe,
                                         LPT_fm, LPT_fpr, LPT_fe]).reshape((1, 15))

        OD_scaling_array_HPT = np.array([Fan_c_fm, Fan_c_fpr, Fan_c_fe,
                                         Fan_d_fm, Fan_d_fpr, Fan_d_fe,
                                         HPC_fm, HPC_fpr, HPC_fe,
                                         HPT_fm + fm, HPT_fpr + pr, HPT_fe + eta,
                                         LPT_fm, LPT_fpr, LPT_fe]).reshape((1, 15))

        OD_scaling_array_LPT = np.array([Fan_c_fm, Fan_c_fpr, Fan_c_fe,
                                         Fan_d_fm, Fan_d_fpr, Fan_d_fe,
                                         HPC_fm, HPC_fpr, HPC_fe,
                                         HPT_fm, HPT_fpr, HPT_fe,
                                         LPT_fm + fm, LPT_fpr + pr, LPT_fe + eta]).reshape((1, 15))
        if scaling_type == 'Fanc':
            return OD_scaling_array_Fanc

        elif scaling_type == 'Fand':
            return OD_scaling_array_Fand

        elif scaling_type == 'HPC':
            return OD_scaling_array_HPC

        elif scaling_type == 'HPT':
            return OD_scaling_array_HPT

        elif scaling_type == 'LPT':
            return OD_scaling_array_LPT

    # create OD arrays

    OD_scaling_array_og = get_OD_scaling_array(fm=0, pr=0, eta=0)

    OD_scaling_array_fm_up_5 = get_OD_scaling_array(fm=0.05, pr=0, eta=0)
    OD_scaling_array_fm_up_15 = get_OD_scaling_array(fm=0, pr=0, eta=0)
    OD_scaling_array_fm_down_5 = get_OD_scaling_array(fm=-0.05, pr=0, eta=0)
    OD_scaling_array_fm_down_15 = get_OD_scaling_array(fm=-0, pr=0, eta=0)

    OD_scaling_array_fpr_up_5 = get_OD_scaling_array(fm=0, pr=0.05, eta=0)
    OD_scaling_array_fpr_up_15 = get_OD_scaling_array(fm=0, pr=0, eta=0)
    OD_scaling_array_fpr_down_5 = get_OD_scaling_array(fm=0, pr=-0.05, eta=0)
    OD_scaling_array_fpr_down_15 = get_OD_scaling_array(fm=0, pr=-0, eta=0)

    OD_scaling_array_fe_up_5 = get_OD_scaling_array(fm=0, pr=0, eta=0.05)
    OD_scaling_array_fe_up_15 = get_OD_scaling_array(fm=0, pr=0, eta=0)
    OD_scaling_array_fe_down_5 = get_OD_scaling_array(fm=0, pr=0, eta=-0.05)
    OD_scaling_array_fe_down_15 = get_OD_scaling_array(fm=0, pr=0, eta=-0)

    # concatenate all OD arrays
    OD_scaling_array = np.concatenate((OD_scaling_array_og,
                                       OD_scaling_array_fm_up_5, OD_scaling_array_fm_up_15, OD_scaling_array_fm_down_5,
                                       OD_scaling_array_fm_down_15,
                                       OD_scaling_array_fpr_up_5, OD_scaling_array_fpr_up_15,
                                       OD_scaling_array_fpr_down_5, OD_scaling_array_fpr_down_15,
                                       OD_scaling_array_fe_up_5, OD_scaling_array_fe_up_15, OD_scaling_array_fe_down_5,
                                       OD_scaling_array_fe_down_15), axis=0)

    print(GEnx_OD)
    # create input array
    input_array = np.concatenate((GEnx_OD, OD_scaling_array), axis=1)
    return input_array


from my_modified_functions import gspdll


def run_simulation(input, save_name):
    y_sim_valid = []
    PRs = []
    ETAs = []
    Ws = []
    TR = []
    FN = []
    Tt = []
    Pt = []
    Nc = []
    Beta = []
    Mach = []

    gspdll.InitializeModel()
    for i in range(len(input)):
        y_sim = np.array(runGsp(gspdll, input[i:i + 1], output_list))
        y_sim_valid.append(y_sim[:, :6])
        # Reynolds = y_sim[:, 6:16]
        PRs.append(y_sim[:, 16:25])
        ETAs.append(y_sim[:, 25:30])
        Ws.append(y_sim[:, 30:39])
        TR.append(y_sim[:, 39:47])
        FN.append(y_sim[:, 47:50])
        Tt.append(y_sim[:, 50:57])
        Pt.append(y_sim[:, 57:64])
        Nc.append(y_sim[:, 64:68])
        Beta.append(y_sim[:, 68:73])
        Mach.append(y_sim[:, 73:78])
    print(Pt)
    print(Beta)
    print(Mach)

    pickle.dump([y_sim_valid, PRs, ETAs, Ws, TR, FN, Tt, Pt, Nc, Beta, Mach],
                open(f"Results/{save_name}.p",
                     "wb"))

run_simulation(get_input(scaling_type='Fanc'), save_name="OD_scaling_Fanc")
run_simulation(get_input(scaling_type='Fand'), save_name="OD_scaling_Fand")
run_simulation(get_input(scaling_type='HPC'), save_name="OD_scaling_HPC")
run_simulation(get_input(scaling_type='HPT'), save_name="OD_scaling_HPT")
run_simulation(get_input(scaling_type='LPT'), save_name="OD_scaling_LPT")

# for i in range(len(run1[0])):  # 0: takeoff 1:climb 2:cruise
#     plt.xlabel('Corrected Fan Speed [%]')
#     plt.ylabel(str(i))
#     plt.legend()
#     plt.show()

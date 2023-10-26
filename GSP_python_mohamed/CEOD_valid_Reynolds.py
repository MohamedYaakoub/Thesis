import pickle
import numpy as np
from GSP_helper import cleanup, runGsp
from matplotlib import pyplot as plt
from map_functions import reset_maps
import pyxdsm.XDSM
# reset_maps()

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
# GSPfileName = "OffDesignGEnx Valid_Shivan.mxl"  # "GEnx-1B_V3_test2.mxl"  #
GSPfileName = "OffDesignGEnx_Valid_All parameters.mxl"  # this is only used for plotting

# file_name = "CEOD_set_Valid.P"
# GEnx_OD, GEnx_OD_true, N1cCEOD = pickle.load(open("CEOD_GEnx/" + file_name, "rb"))
# _, All_Reynolds = pickle.load(open("Constants/Reynolds_" + file_name.strip("CEOD_"), "rb"))

file_name = "CEOD_data_mohamed_2019_feb_1-9_2.p"
GEnx_OD, GEnx_OD_true, N, alt_time, All_Reynolds = pickle.load(open("Reynolds_pickle/Reynolds_" + file_name, "rb"))
Re2_DP, Re25_DP, Re3_DP, Re4_DP, Re49_DP, Re5_DP, Re14_DP, Re19_DP, _, _ = All_Reynolds[0][0, :].T

from my_modified_functions import gspdll

OG_input_list = ["N1", "P0", "T0", "Mach", "HP"]
OD_input_list = ["ODSF_Wc core", "ODSF_PR core", "ODSF_Eta core",  # Fan core
                 "ODSF_Wc duct", "ODSF_PR duct", "ODSF_Eta duct",  # Fan duct
                 "ODSF_Wc", "ODSF_PR", "ODSF_Eta",  # HPC
                 "ODSF_Wc", "ODSF_PR", "ODSF_Eta",  # HPT
                 "ODSF_Wc", "ODSF_PR", "ODSF_Eta"]  # LPT
inputs_list = OG_input_list + OD_input_list

output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2",
               "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19", "Re_6", "Re_9",
               "OPR", "PRc_Fan", "PRd_Fan", "PR_HPC", "PR_HPT", "PR_LPT", "PR_d", "PR_nozzle", "PR_bypass",
               "ETAc_Fan", "ETAd_Fan", "Eta_HPC", "Eta_HPT", "Eta_LPT",
               "W14", "W25", "W4", "Wc2", "Wcc_fan", "Wcd_Fan", "Wc25", "Wc4", "Wc49",
               "TRc_fan", "TRd_fan", "TR_HPC", "TR_HPT", "TR_LPT", "TR_d", "TR_nozzle", "TR_bypass",
               "FN", "FG_nozzle", "FG_bypass",
               "Tt2", "Tt14", "Tt25", "Tt3", "Tt4", "Tt49", "Tt5",
               "Pt2", "Pt14", "Pt25", "Pt3", "Pt4", "Pt49", "Pt5"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))

DP_calibration = True  # This for takeoff calibration
splines = True  # This to activate splines
OD_scaling = True  # This to activate OD scaling

if DP_calibration:
    SF_design_point_TO = [0.99802762, 0.95000085, 0.99649734, 0.99076545, 0.99791594, 1.02033115,
                          0.97803967, 0.97923359]
else:
    SF_design_point_TO = [1] * 8

if OD_scaling:
    X = [-0.02223825, 0.09164879, -0.09991125, 0.19766541, -0.00923643, 0.0319769,
         - 0.09198503, 0.0346309, 0.04168849, -0.19995974, 0.01842454, -0.09209377,
         - 0.04210804, 0.0318016, -0.09989942, -0.10127006]
    if splines:
        X_takeoff = [0] * 16

        X_climb = [-0.018920801466503236, 0.08026750589244025, -0.09992898454200377, 0.19997674049206704,
                   -0.00855507010666653, 0.015661473372625537, -0.06654936228200746, 0.09996060229972696,
                   0.036674812908995526, -0.1929803759384588, 0.018529168900401904, -0.07614862046892305,
                   -0.06336451991859186, -0.02078831028334359, -0.09202354759620704, -0.07207633859121175]

        X_cruise = [-0.02223825, 0.09164879, -0.09991125, 0.19766541, -0.00923643, 0.0319769,
         - 0.09198503, 0.0346309, 0.04168849, -0.19995974, 0.01842454, -0.09209377,
         - 0.04210804, 0.0318016, -0.09989942, -0.10127006]
    else:
        X_takeoff = X
        X_climb = X
        X_cruise = X
else:
    X = [0] * 22
    X_takeoff = X
    X_climb = X
    X_cruise = X


def scaling_F(ReDP, ReOD, a, b, initial_value=1):
    """
    Scaling function is a second degree polynomial
    :param ReDP: design spool speed
    :param ReOD: off-design spool speed
    :return: function value
    """
    return np.array(initial_value + a * ((ReOD - ReDP) / ReDP) + b * ((ReOD - ReDP) / ReDP) ** 2)


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


def scale_maps_reynolds(typef, ReDP, ReOD, file_name, poly_param, initial_values):
    X = poly_param

    if typef == 'C':
        fm_initial_value = initial_values[0]
        fe_initial_value = initial_values[1]

        fm = scaling_F(ReDP, ReOD, X[0], X[1], initial_value=fm_initial_value)
        fe = scaling_F(ReDP, ReOD, X[2], X[3], initial_value=fe_initial_value)
        return fm, fe
    else:
        fe_initial_value = initial_values[0]
        fe = scaling_F(ReDP, ReOD, X[0], X[1], initial_value=fe_initial_value)
        return fe


def simulate(Re, inputDat, X):
    y_sim = []

    Re2_i, Re25_i, Re3_i, Re4_i, Re49_i, Re5_i, Re14_i, Re19_i, _, _ = Re.T

    for i in range(len(inputDat)):
        # Fan_c_fm, Fan_c_fe, Fan_d_fm, Fan_d_fe, HPC_fm, HPC_fe, HPT_fe, LPT_fe = SFDP

        Fan_c_fm, Fan_c_fe = \
            scale_maps_reynolds("C", Re2_DP, Re2_i[i], "1_LPC_core", X[:4], initial_values=SF_design_point_TO[0:2])
        Fan_d_fm, Fan_d_fe = \
            scale_maps_reynolds("C", Re2_DP, Re2_i[i], "2_LPC_bypass", X[4:8], initial_values=SF_design_point_TO[2:4])
        HPC_fm, HPC_fe = \
            scale_maps_reynolds("C", Re25_DP, Re25_i[i], "3_HPC", X[8:12], initial_values=SF_design_point_TO[4:6])
        HPT_fe = scale_maps_reynolds("T", Re4_DP, Re4_i[i], "4_HPT", X[12:14], initial_values=SF_design_point_TO[6:7])
        LPT_fe = scale_maps_reynolds("T", Re49_DP, Re49_i[i], "5_LPT", X[14:16], initial_values=SF_design_point_TO[7:8])

        # if DP_calibration:
        #     if Take_off_switch:
        #         Fan_c_fm, Fan_c_fe, Fan_d_fm, Fan_d_fe, HPC_fm, HPC_fe, HPT_fe, LPT_fe = SF_design_point_TO

        OD_scaling_array = np.array([Fan_c_fm, 1, Fan_c_fe,
                                     Fan_d_fm, 1, Fan_d_fe,
                                     HPC_fm, 1, HPC_fe,
                                     1, 1, HPT_fe,
                                     1, 1, LPT_fe])
        # print(OD_scaling_array)
        print("OD scaling array", OD_scaling_array)
        run_gsp_input = np.concatenate((inputDat[i, :], OD_scaling_array), axis=0)
        y_sim_iter = runGsp(gspdll, run_gsp_input, output_list)
        y_sim.append(y_sim_iter)
    return np.array(y_sim)


# %%
def compute_error(inputDat, trueVal, Re, X_array):
    y_sim = simulate(Re, inputDat, X_array)
    y_sim_valid = y_sim[:, :6]
    Reynolds = y_sim[:, 6:16]
    PRs = y_sim[:, 16:25]
    ETAs = y_sim[:, 25:30]
    Ws = y_sim[:, 30:39]
    TR = y_sim[:, 39:47]
    FN = y_sim[:, 47:50]
    Tt = y_sim[:, 50:57]
    Pt = y_sim[:, 57:64]
    Nc = y_sim[:, 64:68]

    change = (trueVal - y_sim_valid) / (trueVal + 0.000001)
    meanE = 100 * np.sqrt(np.mean(change ** 2, axis=0))
    # return meanE, change * 100, np.mean(efficiencies, axis=0), Reynolds
    return meanE, change * 100, y_sim_valid, Reynolds, PRs, ETAs, Ws, TR, FN, Tt, Pt, Nc


def run_validation():
    paramE = output_list[:6]
    counter = 0

    All_change = []
    All_validation_params = []
    ALL_PRs = []
    All_ETAs = []
    All_Ws = []
    All_TR = []
    All_Fn = []
    All_Tt = []
    All_Pt = []
    All_Nc = []

    meanL = []
    stdL = []
    stdLR = []
    gspdll.InitializeModel()
    for i, j, k in zip(GEnx_OD, GEnx_OD_true, All_Reynolds):

        if counter == 0:
            print("take off start")
            X = X_takeoff
            Take_off_switch = True

        elif counter == 1:
            print("climb start")
            X = X_climb
            Take_off_switch = False

        else:
            print("cruise start")
            X = X_cruise
            Take_off_switch = False

        counter += 1
        print("counter value: ", counter)
        print('Loop initiated')
        print(X)

        inputDat = i
        trueVal = j
        Re = k

        mean, change, validation_params, Reynolds, PRs, ETAs, Ws, TR, FN, Tt, Pt, Nc = compute_error(inputDat, trueVal, Re,
                                                                                                 X)

        print('RE', Reynolds)
        print('PRs', PRs)
        print('eta', ETAs)
        print('ws', Ws)
        print('Tr', TR)
        print('FN', FN)
        meanL.append(list(mean))
        stdL.append(list(np.std(change, axis=0)))
        # EtaL.append(list(Eta * 100))
        All_change.append(change / 100)

        All_validation_params.append(validation_params)
        ALL_PRs.append(PRs)
        All_ETAs.append(ETAs)
        All_Ws.append(Ws)
        All_TR.append(TR)
        All_Fn.append(FN)
        All_Tt.append(Tt)
        All_Pt.append(Pt)
        All_Nc.append(Nc)

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

    # barC(EtaL, ['Take-off', 'Climb', 'Cruise'], Etas, "Efficiency [%]")
    All_change = [item for sublist in All_change for item in sublist]
    Rms = np.sqrt(np.mean(np.mean(np.array(All_change) ** 2, axis=0)))
    if OD_scaling:
        barC(meanL, ['Take-off', 'Climb', 'Cruise'], paramE, "Error [%]",
             f'{file_name.strip("CEOD_").strip(".p")} \n with Re Correction \n RMSE: {str(round(Rms, 6))}')
    else:
        barC(meanL, ['Take-off', 'Climb', 'Cruise'], paramE, "Error [%]",
             f'{file_name.strip("CEOD_").strip(".p")} \n No Re Correction \n RMSE: {str(round(Rms, 6))}')
    print(Rms, "rms")
    print("Part 1 done")

    if DP_calibration:
        if splines:
            pickle.dump([All_validation_params, ALL_PRs, All_ETAs, All_Ws, All_TR, All_Fn, All_Tt, All_Pt],
                        open("Results/Results2_splines_DP_calibration_" + file_name.strip("CEOD_"), "wb"))
        else:
            if OD_scaling:
                pickle.dump([All_validation_params, ALL_PRs, All_ETAs, All_Ws, All_TR, All_Fn, All_Tt, All_Pt],
                            open("Results/Results_one_equation_DP_calibration_" + file_name.strip("CEOD_"), "wb"))
            else:
                pickle.dump([All_validation_params, ALL_PRs, All_ETAs, All_Ws, All_TR, All_Fn, All_Tt, All_Pt],
                            open(
                                "Results/Results_no_OD_scaling_DP_calibration_" + file_name.strip("CEOD_"),
                                "wb"))
    else:
        if splines:
            pickle.dump([All_validation_params, ALL_PRs, All_ETAs, All_Ws, All_TR, All_Fn, All_Tt, All_Pt],
                        open("Results/Results_splines_" + file_name.strip("CEOD_"), "wb"))
        else:
            if OD_scaling:
                pickle.dump([All_validation_params, ALL_PRs, All_ETAs, All_Ws, All_TR, All_Fn, All_Tt, All_Pt],
                            open("Results/Results_one_equation_" + file_name.strip("CEOD_"), "wb"))
            else:
                pickle.dump([All_validation_params, ALL_PRs, All_ETAs, All_Ws, All_TR, All_Fn, All_Tt, All_Pt],
                            open("Results/Results_no_OD_scaling_" + file_name.strip("CEOD_"), "wb"))


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


if __name__ == '__main__':
    run_validation()
    # cleanup(gspdll)

# %%

import pickle
import numpy as np
from GSP_helper import cleanup, runGsp
from matplotlib import pyplot as plt
from map_functions import reset_maps

reset_maps()

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
# GSPfileName = "OffDesignGEnx Valid_Shivan.mxl"  # "GEnx-1B_V3_test2.mxl"  #
GSPfileName = "OffDesignGEnx OD Scaling.mxl"  # this is only used for plotting

# file_name = "CEOD_set_Valid.P"
# GEnx_OD, GEnx_OD_true, N1cCEOD = pickle.load(open("CEOD_GEnx/" + file_name, "rb"))
# _, All_Reynolds = pickle.load(open("Constants/Reynolds_" + file_name.strip("CEOD_"), "rb"))

file_name = "CEOD_data_mohamed_2019_feb_1-9_1.p"
GEnx_OD, GEnx_OD_true, N, alt_time, All_Reynolds = pickle.load(open("Reynolds_pickle/Reynolds_" + file_name, "rb"))

Re2_DP, Re25_DP, Re3_DP, Re4_DP, Re49_DP, Re5_DP, Re14_DP, Re19_DP, _, _ = All_Reynolds[0][0, :].T

from my_modified_functions import gspdll

inputs_list = ["N1", "P0", "T0", "Mach", "HP"]
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2",
               "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19",
               "OPR", "PRc_Fan", "PRd_Fan", "PR_HPC", "PR_HPT", "PR_LPT",
               "ETAc_Fan", "ETAd_Fan", "Eta_HPC", "Eta_HPT", "Eta_LPT",
               "W14", "W25", "W4"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))

X = [0.03108588422207266, 0.02567619660579215, 0.007149626032918022, 0.017087814396480966,
     -0.09674097350611965, 0.013906217694496653, -0.09455849175478899, 0.00828532128394946,
     0.042652297505362174, -0.0081375030413752, 0.05059589749640184, -0.0007653113564509151,
     0.021004053043303397, 0.0402277436164645, 0.03290261764660938, 0.01442539496720846,
     -0.0700287864467976, -0.13871223330448718, -0.046010533084381904, 0.02678507938839153,
     -0.06295842712479592, -0.11441280321325185]

# X_takeoff = X
X_climb = X
X_cruise = X

splines = True

X_takeoff = [0] * 22
#
# X_climb = [-0.39151153304030584, -0.3572569487085121, -0.11508814965214664, 0.6040357595244259, -0.5473006373047256,
#            -0.4337901143312386, -0.9781608146532708, 0.9872048995550504, -0.9250162770160874, -0.3300894211317793,
#            -0.7196445323625377, -0.63761329701412, -0.37761050930438855, 0.37409468505190535, -0.7169121028867804,
#            0.504095168408869, 0.004084270145224167, 0.5007723795343721, -0.0414428972840476, 0.47249205746563394,
#            -0.7884995578550504, -0.2725844710128533]
#
# X_cruise = [0.57412476, -0.90224694, 0.08398058, -0.30101288, -0.04481059, -0.10244869, 0.4126201, 0.26567765,
#             -0.5879205, -0.29147754, 0.65966397, -0.28873865, -0.16545684, 0.63965032, -0.9634026, 0.65153249,
#             -0.00253814, -0.84644459, -0.21551771, 0.20316737, -0.94047361, -0.05217237]
# splines = True





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
        fm = scaling_F(ReDP, ReOD, X[2], X[3])
        fpr = scaling_F(ReDP, ReOD, X[0], X[1])
        fe = scaling_F(ReDP, ReOD, X[4], X[5])
        return np.clip(fm, 0.8, 1.05), np.clip(fpr, 0.8, 1.05), np.clip(fe, 0.8, 1.05)
    else:
        fe = scaling_F(ReDP, ReOD, X[0], X[1])
        return np.clip(fe, 0.8, 1.05)

def simulate(Re, inputDat, X):
    y_sim = []

    Re2_i, Re25_i, Re3_i, Re4_i, Re49_i, Re5_i, Re14_i, Re19_i, _, _ = Re.T
    # print(Re25_i[0], Re3_i[0], Re49_i[0], Re5_i[0], Re19_i[0])

    for i in range(len(inputDat)):
        Fan_c_fm, Fan_c_fpr, Fan_c_fe = scale_maps_reynolds("C", Re25_DP, Re25_i[i], "1_LPC_core", X[:6])
        Fan_d_fm, Fan_d_fpr, Fan_d_fe = scale_maps_reynolds("C", Re19_DP, Re19_i[i], "2_LPC_bypass", X[6:12])
        HPC_fm, HPC_fpr, HPC_fe = scale_maps_reynolds("C", Re3_DP, Re3_i[i], "3_HPC", X[12:18])
        HPT_fe = scale_maps_reynolds("T", Re49_DP, Re49_i[i], "4_HPT", X[18:20])
        LPT_fe = scale_maps_reynolds("T", Re5_DP, Re5_i[i], "5_LPT", X[20:22])

        OD_scaling_array = np.array([Fan_c_fm, Fan_c_fpr, Fan_c_fe,
                                     Fan_d_fm, Fan_d_fpr, Fan_d_fe,
                                     HPC_fm, HPC_fpr, HPC_fe,
                                     1, 1, HPT_fe,
                                     1, 1, LPT_fe])
        # print(OD_scaling_array)

        run_gsp_input = np.concatenate((inputDat[i, :], OD_scaling_array), axis=0)
        y_sim_iter = runGsp(gspdll, run_gsp_input, output_list)
        y_sim_iter = y_sim_iter[:6]  # ignore effs for now
        y_sim.append(y_sim_iter)

    return np.array(y_sim)


# %%
def compute_error(inputDat, trueVal, Re, X_array):
    y_sim = simulate(Re, inputDat, X_array)
    y_sim_valid = y_sim[:, :6]
    # efficiencies = y_sim[:, 6:]
    Reynolds = y_sim[:, 6:14]
    PRs = y_sim[:, 14:20]
    ETAs = y_sim[:, 20:25]
    Ws = y_sim[:, 25:]

    change = (trueVal - y_sim_valid) / (trueVal + 0.000001)
    meanE = 100 * np.sqrt(np.mean(change ** 2, axis=0))
    # return meanE, change * 100, np.mean(efficiencies, axis=0), Reynolds
    return meanE, change * 100, Reynolds, PRs, ETAs, Ws


meanL = []
stdL = []
stdLR = []
# EtaL = []


# Re_4 = []

# All_Reynolds = []
ALL_PRs = []
All_ETAs = []
All_Ws = []


def run_validation():
    paramE = output_list[:6]

    All_change = []
    counter = 0
    for i, j, k in zip(GEnx_OD, GEnx_OD_true, All_Reynolds):

        if counter == 0:
            print("take off start")
            X = X_takeoff
        elif counter == 1:
            print("climb start")
            X = X_climb
        else:
            print("cruise start")
            X = X_cruise

        counter += 1
        print("counter value: ", counter)

        print(X)
        print('Loop initiated')

        inputDat = i
        trueVal = j
        Re = k
        mean, change, Reynolds, PRs, ETAs, Ws = compute_error(inputDat, trueVal, Re, X)
        meanL.append(list(mean))
        stdL.append(list(np.std(change, axis=0)))
        # EtaL.append(list(Eta * 100))
        All_change.append(change / 100)
        ALL_PRs.append(PRs)
        print("PRS", PRs)
        # print("ALL Prs", ALL_PRs)
        All_ETAs.append(ETAs)
        All_Ws.append(Ws)

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
    barC(meanL, ['Take-off', 'Climb', 'Cruise'], paramE, "Error [%]",
         f'{file_name.strip("CEOD_").strip(".p")} \n with Re Correction \n RMSE: {str(round(Rms, 6))}')
    print(Rms, "rms")
    print("Part 1 done")

    if splines:
        pickle.dump([ALL_PRs, All_ETAs, All_Ws], open("Results/Results_splines_" + file_name.strip("CEOD_"), "wb"))
    else:
        pickle.dump([ALL_PRs, All_ETAs, All_Ws], open("Results/Results_one_equation_" + file_name.strip("CEOD_"), "wb"))



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

import pickle
import numpy as np
import timeit
import sys
from GSP_helper import cleanup, runGsp
import subprocess
# from Reynolds_effect_functions import scale_maps_reynolds
from matplotlib import pyplot as plt
from WriteCMapGSP import read_mapC, write_mapC
from WriteTMapGSP import read_mapT, write_mapT

from map_functions import reset_maps

reset_maps()

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
# GSPfileName = "OffDesignGEnx Valid_Shivan.mxl"  # "GEnx-1B_V3_test2.mxl"  #
GSPfileName = "OffDesignGEnx_Valid_All parameters.mxl"  # this is only used for plotting

file_name = "CEOD_set_Valid.P"
# file_name = "CEOD_200408-203904-KLM168____-KATLEHAM-KL_PH-BHA-2-956609-W010FFD.P"
# file_name = "CEOD_160724-193429-KLM891____-EHAMZUUU-KL_PH-BHA-2-956609-W007FFD.p"
# file_name =


GEnx_OD, GEnx_OD_true, N1cCEOD = pickle.load(open("CEOD_GEnx/" + file_name, "rb"))
_, All_Reynolds = pickle.load(open("Constants/Reynolds_" + file_name.strip("CEOD_"), "rb"))

Re2_DP, Re25_DP, Re3_DP, Re4_DP, Re49_DP, Re5_DP, Re14_DP, Re19_DP = All_Reynolds[0][0, :].T

from my_modified_functions import gspdll

inputs_list = ["N1", "P0", "T0", "Mach", "HP"]
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2",
               "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19",
               "OPR", "PRc_Fan", "PRd_Fan", "PR_HPC", "PR_HPT", "PR_LPT",
               "ETAc_Fan", "ETAd_Fan", "Eta_HPC", "Eta_HPT", "Eta_LPT",
               "W14", "W25", "W4"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))

X = [-0.9924531509396756, 0.9493111782594137, 0.015398502958351434, -0.5188458123089683, -0.22308564787498586,
     -0.730856647237857, 0.37248086082138787, 0.27039831352397137, -0.9584642768648018, -0.3295273625460893,
     -0.5905141960366497, -0.6997725432657045, -0.06270267183037614, -0.4289624389993987, 0.05691519027254244,
     -0.18430762512912546, 0.28719233155558155, -0.7437196774085011, -0.61251462659736, 0.28412132635236453,
     -0.8649627124131621, 0.12693395455166834]

X_takeoff = X
X_climb = X
X_cruise = X

splines = False

# X_takeoff = [-0.9983389133736509, 0.9385307347869887, 0.034000861580716135, -0.03577821327586628, -0.9981798259300767,
#              0.9530924154305069, -0.995700235073686, 0.8304600340231876, -0.13591022775450678, -0.11859373577835941,
#              -0.649524267286157, 0.5760310215827391, 0.09742118796651944, 0.1789460709533659, 0.04414420344277481,
#              0.1036752582677658, -0.5609002492275668, 0.8606842835909971, -0.423764318345767, 0.45272003883100775,
#              0.2975860084469224, -0.48489697839506674]
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
        MdotC, EtaC, PRC, surge_mC, surge_pC, NC = pickle.load(open("Constants/" + file_name + "pick.p", "rb"))
        fpr = scaling_F(ReDP, ReOD, X[0], X[1])
        fm = scaling_F(ReDP, ReOD, X[2], X[3])
        fe = scaling_F(ReDP, ReOD, X[4], X[5])

        surgeN = np.unique(NC)

        fsurge_pr = scaling_F(ReDP, surgeN, X[0], X[1])
        fsurge_m = scaling_F(ReDP, surgeN, X[2], X[3])
        fsurge_m = np.insert(fsurge_m, 0, 1)
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
        fe = scaling_F(ReDP, ReOD, X[0], X[1])
        # fpr = np.insert(fpr, 0, 1)

        write_mapT(file_name, file_name, np.clip(PRmin * 1, 0, 100), np.clip(PRmax * 1, 0, 100), np.clip(MdotT * 1,
                                                                                                         0.05, 2000),
                   np.clip(EtaT * fe, 0.10101, 0.99), NT, BT)


def simulate(Re, inputDat, X):
    y_sim = []

    # Re25_DP = 18660698.2305537
    # Re19_DP = 26333708.4295415
    #
    # # Re19_DP = 61973246.778017
    # # Re3_DP = 35472717.4019559
    #
    # Re3_DP = 61973246.778017
    # Re49_DP = 22736943.8967379
    # Re5_DP = 18683167.9100408

    Re2_i, Re25_i, Re3_i, Re4_i, Re49_i, Re5_i, Re14_i, Re19_i = Re.T
    # print(Re25_i[0], Re3_i[0], Re49_i[0], Re5_i[0], Re19_i[0])
    for i in range(len(inputDat)):
        # scale_maps_reynolds("C", Re25_i[0], Re25_i[i], "1_LPC_core", X[:6])
        # scale_maps_reynolds("C", Re19_i[0], Re19_i[i], "2_LPC_bypass", X[6:12])
        # scale_maps_reynolds("C", Re3_i[0], Re3_i[i], "3_HPC", X[12:18])
        # scale_maps_reynolds("T", Re49_i[0], Re49_i[i], "4_HPT", X[18:20])
        # scale_maps_reynolds("T", Re5_i[0], Re5_i[i], "5_LPT", X[20:22])
        scale_maps_reynolds("C", Re25_DP, Re25_i[i], "1_LPC_core", X[:6])
        scale_maps_reynolds("C", Re19_DP, Re19_i[i], "2_LPC_bypass", X[6:12])
        scale_maps_reynolds("C", Re3_DP, Re3_i[i], "3_HPC", X[12:18])
        scale_maps_reynolds("T", Re49_DP, Re49_i[i], "4_HPT", X[18:20])
        scale_maps_reynolds("T", Re5_DP, Re5_i[i], "5_LPT", X[20:22])
        gspdll.InitializeModel()

        y_sim_iter = np.array(runGsp(gspdll, inputDat[i], output_list))
        # y_sim_iter = y_sim_iter[:6]  # ignore effs for now
        y_sim.append(y_sim_iter)

    y_sim = np.array(y_sim)

    return y_sim


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

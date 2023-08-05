import pickle
import numpy as np
import os
from matplotlib import pyplot as plt

# X = [-0.9924531509396756, 0.9493111782594137, 0.015398502958351434,
#      -0.5188458123089683, -0.22308564787498586, -0.730856647237857,
#      0.37248086082138787, 0.27039831352397137, -0.9584642768648018,
#      -0.3295273625460893, -0.5905141960366497, -0.6997725432657045,
#      -0.06270267183037614, -0.4289624389993987, 0.05691519027254244,
#      -0.18430762512912546, 0.28719233155558155, -0.7437196774085011,
#      -0.61251462659736, 0.28412132635236453, -0.8649627124131621,
#      0.12693395455166834]

X = [-0.9924531509396756, 0.9493111782594137, 0.015398502958351434, -0.5188458123089683, -0.22308564787498586,
     -0.730856647237857, 0.37248086082138787, 0.27039831352397137, -0.9584642768648018, -0.3295273625460893,
     -0.5905141960366497, -0.6997725432657045, -0.06270267183037614, -0.4289624389993987, 0.05691519027254244,
     -0.18430762512912546, 0.28719233155558155, -0.7437196774085011, -0.61251462659736, 0.28412132635236453,
     -0.8649627124131621, 0.12693395455166834]
X_takeoff = X
X_climb = X
X_cruise = X

splines = False


#
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


file_name = "CEOD_set_Valid.P"
# file_name = "CEOD_200408-203904-KLM168____-KATLEHAM-KL_PH-BHA-2-956609-W010FFD.P"
# file_name = "CEOD_160724-193429-KLM891____-EHAMZUUU-KL_PH-BHA-2-956609-W007FFD.p"


GEnx_OD, GEnx_OD_true, N1cCEOD = pickle.load(open("CEOD_GEnx/" + file_name, "rb"))
_, All_Reynolds = pickle.load(open("Constants/Reynolds_" + file_name.strip("CEOD_"), "rb"))
if splines:
    ALL_PRs, All_ETAs, All_Ws = pickle.load(open("Results/Results_" + file_name.strip("CEOD_"), "rb"))
else:
    ALL_PRs, All_ETAs, All_Ws = pickle.load(open("Results/Results_one_equation_" + file_name.strip("CEOD_"), "rb"))
    print("one equation")


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")

if splines:
    directory_path = "C:/Users/mohsy/University/KLM/Thesis/My thesis/Plots/" + file_name.strip("CEOD_").strip(
        ".P") + "_splines"
else:
    directory_path = "C:/Users/mohsy/University/KLM/Thesis/My thesis/Plots/" + file_name.strip("CEOD_").strip(
        ".P") + "_one_equation"

create_directory(directory_path)

for i, flight_cond in enumerate(["Take Off", "Climb", "Cruise"]):
    OPR, PRc_Fan, PRd_Fan, PR_HPC, PR_HPT, PR_LPT = ALL_PRs[i].T
    ETAc_Fan, ETAd_Fan, ETA_HPC, ETA_HPT, ETA_LPT = All_ETAs[i].T
    W_14, W_25, W_4 = All_Ws[i].T

    _, Re25_DP, Re3_DP, _, Re49_DP, Re5_DP, _, Re19_DP = All_Reynolds[0].T[:, 0]

    # Re19_DP = Re19[0]
    # Re25_DP = Re25[0]
    # Re3_DP = Re3[0]
    # Re49_DP = Re4[0]
    # Re5_DP = Re5[0]

    Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19 = All_Reynolds[i].T
    N1 = GEnx_OD[i][:, 0]


    def tm_plot(y_array, tm_type, x_iters, polynomials, re_dp, re):
        fig, ax1 = plt.subplots(2, figsize=(6, 6), dpi=300)
        # tm_type = "HPC"
        for j, y_axis in enumerate(["PR", "Efficiency"]):
            # y_array = [PR_HPC, ETA_HPC]
            color = 'tab:red'
            ax1[j].set_xlabel("N1")
            ax1[j].set_ylabel(y_axis + "_" + tm_type, color=color)
            ax1[j].scatter(N1, y_array[j], color=color)
            ax1[j].tick_params(axis='y', labelcolor=color)
            # ax1[j].set_title(flight_cond)

            k = 2
            ax1[j].grid()
            ax2 = ax1[j].twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            if tm_type[-1] == "T":
                y_axis = "Efficiency"
            ax2.set_ylabel(f'SF_{y_axis}_{tm_type}',
                           color=color)  # we already handled the x-label with ax1
            x_iter = None
            if tm_type[-1] == "T":
                x_iter = x_iters[0]
            elif y_axis == "PR":
                x_iter = x_iters[0]
            elif y_axis == "Efficiency":
                x_iter = x_iters[1]

            ax2.scatter(N1, scaling_F(re_dp, re, polynomials[x_iter], polynomials[x_iter + 1]), color=color)
            ax2.tick_params(axis='y', labelcolor=color)

        fig.suptitle(tm_type + " " + flight_cond)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(directory_path + "/" + tm_type + " " + flight_cond + ".jpg")
        plt.show()


    def Mass_flow_plots():
        for mass_flow, label, title in zip([W_14, W_25, W_4],
                                           ["W_14", "W_25", "W_4"],
                                           ["Bypass", "Core - fwd combustor", "Core - aft combustor"]):
            fig = plt.figure(figsize=(6, 3), dpi=300)
            plt.scatter(N1, mass_flow)
            plt.ylabel(f'{label} kg/s')
            plt.xlabel('N1')
            plt.suptitle(title + " " + flight_cond)
            plt.grid()
            plt.tight_layout()
            plt.savefig(directory_path + "/" + title + " " + flight_cond + ".jpg")
            plt.show()

            # fig, ax1 = plt.subplots(2, figsize=(6, 6), dpi=300)
            # color = 'tab:red'
            # ax1.set_xlabel("N1")
            # ax1.ylabel(f'{label} kg/s', color=color)
            # ax1.scatter(N1, y_array[j], color=color)
            # ax1.tick_params(axis='y', labelcolor=color)
            #
            # ax1.grid()
            # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            #
            # color = 'tab:blue'
            # ax2.set_ylabel((f'{label} kg/s'),
            #                color=color)  # we already handled the x-label with ax1
            # ax2.scatter(N1, scaling_F(re_dp, re, polynomials[x_iter], polynomials[x_iter + 1]), color=color)
            # ax2.tick_params(axis='y', labelcolor=color)

            # fig.suptitle(title + " " + flight_cond)
            # fig.tight_layout()  # otherwise the right y-label is slightly clipped
            # plt.savefig(directory_path + "/" + title + " " + flight_cond + ".jpg")
            # plt.show()


    if i == 0:
        print("take off start")
        X = X_takeoff
    elif i == 1:
        print("climb start")
        X = X_climb
    else:
        print("cruise start")
        X = X_cruise

    LPCc_X = X[:6]
    LPCd_X = X[6:12]
    HPC_X = X[12:18]
    HPT_X = X[18:20]
    LPT_X = X[20:22]

    tm_plot(y_array=[PR_HPC, ETA_HPC], tm_type="HPC", x_iters=[0, 4], polynomials=HPC_X, re_dp=Re3_DP, re=Re3)
    tm_plot(y_array=[PRc_Fan, ETAc_Fan], tm_type="LPCc", x_iters=[0, 4], polynomials=LPCc_X, re_dp=Re25_DP, re=Re25)
    tm_plot(y_array=[PRd_Fan, ETAd_Fan], tm_type="LPCd", x_iters=[0, 4], polynomials=LPCd_X, re_dp=Re19_DP, re=Re19)
    tm_plot(y_array=[PR_HPT, ETA_HPT], tm_type="HPT", x_iters=[0], polynomials=HPT_X, re_dp=Re49_DP, re=Re49)
    tm_plot(y_array=[PR_LPT, ETA_LPT], tm_type="LPT", x_iters=[0], polynomials=LPT_X, re_dp=Re5_DP, re=Re5)
    Mass_flow_plots()

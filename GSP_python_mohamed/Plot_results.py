import pickle
import numpy as np
import os
from matplotlib import pyplot as plt
#
iter_time, iter_Xi = pickle.load(open("New solves OD scaling/Nozzle single equation iter1.p", "rb"))
print(iter_Xi)

# X = [0.04995546, 0.02878598, -0.00712323, 0.01510955, -0.08778622, -0.05025517,
#      - 0.09904944, 0.04936974, 0.03271384, -0.01814827, 0.05167898, 0.03520765,
#      0.02111619, 0.04719428, 0.03002022, -0.00058276, -0.05902952, -0.14813571,
#      -0.03774783, 0.04783216, -0.0081694, -0.04824031]
#
# X_takeoff = X
# X_climb = X
# X_cruise = X
#
# splines = False
#
#
# # splines = True
#
# def scaling_F(ReDP, ReOD, a, b):
#     """
#     Scaling function is a second degree polynomial
#     :param ReDP: design spool speed
#     :param ReOD: off-design spool speed
#     :return: function value
#     """
#     return np.array(1 + a * ((ReOD - ReDP) / ReDP) + b * ((ReOD - ReDP) / ReDP) ** 2)
#
#
# # file_name = "CEOD_set_Valid.P"
# GSPfileName = "OffDesignGEnx_Valid_All parameters.mxl"
#
#
# file_name = "CEOD_data_mohamed_2019_feb_1-9_1.p"
#
# GEnx_OD, GEnx_OD_true, N, alt_time, All_Reynolds = pickle.load(open("Reynolds_pickle/Reynolds_" + file_name, "rb"))
# Re2_DP, Re25_DP, Re3_DP, Re4_DP, Re49_DP, Re5_DP, Re14_DP, Re19_DP, _, _ = All_Reynolds[0][0, :].T
#
#
# if splines:
#     All_Valid_params, ALL_PRs, All_ETAs, All_Ws = pickle.load(open("Results/Results_" + file_name.strip("CEOD_"), "rb"))
# else:
#     All_Valid_params, ALL_PRs, All_ETAs, All_Ws = pickle.load(open("Results/Results_one_equation_" + file_name.strip("CEOD_"), "rb"))
#     print("one ALL_PRs")
#
#
# def create_directory(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#         print(f"Directory '{directory}' created successfully.")
#     else:
#         print(f"Directory '{directory}' already exists.")
#
#
# if splines:
#     directory_path = "C:/Users/mohsy/University/KLM/Thesis/My thesis/Plots/" + file_name.strip("CEOD_").strip(
#         ".P") + "_splines"
# else:
#     directory_path = "C:/Users/mohsy/University/KLM/Thesis/My thesis/Plots/" + file_name.strip("CEOD_").strip(
#         ".P") + "_one_equation"
#
# create_directory(directory_path)
#
# for i, flight_cond in enumerate(["Take Off", "Climb", "Cruise"]):
#     OPR, PRc_Fan, PRd_Fan, PR_HPC, PR_HPT, PR_LPT = ALL_PRs[i].T
#     ETAc_Fan, ETAd_Fan, ETA_HPC, ETA_HPT, ETA_LPT = All_ETAs[i].T
#     W_14, W_25, W_4 = All_Ws[i].T
#     # Re19_DP = Re19[0]
#     # Re25_DP = Re25[0]
#     # Re3_DP = Re3[0]
#     # Re49_DP = Re4[0]
#     # Re5_DP = Re5[0]
#
#     Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19, _, _ = All_Reynolds[i].T
#     N1 = GEnx_OD[i][:, 0]
#
#
#     def tm_plot(y_array, tm_type, x_iters, polynomials, re_dp, re):
#         fig, ax1 = plt.subplots(2, figsize=(6, 6), dpi=300)
#         # tm_type = "HPC"
#         for j, y_axis in enumerate(["PR", "Efficiency"]):
#             # y_array = [PR_HPC, ETA_HPC]
#             color = 'tab:red'
#             ax1[j].set_xlabel("N1")
#             ax1[j].set_ylabel(y_axis + "_" + tm_type, color=color)
#             ax1[j].scatter(N1, y_array[j], color=color)
#             ax1[j].tick_params(axis='y', labelcolor=color)
#             # ax1[j].set_title(flight_cond)
#
#             k = 2
#             ax1[j].grid()
#             ax2 = ax1[j].twinx()  # instantiate a second axes that shares the same x-axis
#             color = 'tab:blue'
#             if tm_type[-1] == "T":
#                 y_axis = "Efficiency"
#             ax2.set_ylabel(f'SF_{y_axis}_{tm_type}',
#                            color=color)  # we already handled the x-label with ax1
#             x_iter = None
#             if tm_type[-1] == "T":
#                 x_iter = x_iters[0]
#             elif y_axis == "PR":
#                 x_iter = x_iters[0]
#             elif y_axis == "Efficiency":
#                 x_iter = x_iters[1]
#
#             ax2.scatter(N1, scaling_F(re_dp, re, polynomials[x_iter], polynomials[x_iter + 1]), color=color)
#             ax2.tick_params(axis='y', labelcolor=color)
#
#         fig.suptitle(tm_type + " " + flight_cond)
#         fig.tight_layout()  # otherwise the right y-label is slightly clipped
#         plt.savefig(directory_path + "/" + tm_type + " " + flight_cond + ".jpg")
#         plt.show()
#
#
#     def Mass_flow_plots():
#         for mass_flow, label, title in zip([W_14, W_25, W_4],
#                                            ["W_14", "W_25", "W_4"],
#                                            ["Bypass", "Core - fwd combustor", "Core - aft combustor"]):
#             fig = plt.figure(figsize=(6, 3), dpi=300)
#             plt.scatter(N1, mass_flow)
#             plt.ylabel(f'{label} kg/s')
#             plt.xlabel('N1')
#             plt.suptitle(title + " " + flight_cond)
#             plt.grid()
#             plt.tight_layout()
#             plt.savefig(directory_path + "/" + title + " " + flight_cond + ".jpg")
#             plt.show()
#
#             # fig, ax1 = plt.subplots(2, figsize=(6, 6), dpi=300)
#             # color = 'tab:red'
#             # ax1.set_xlabel("N1")
#             # ax1.ylabel(f'{label} kg/s', color=color)
#             # ax1.scatter(N1, y_array[j], color=color)
#             # ax1.tick_params(axis='y', labelcolor=color)
#             #
#             # ax1.grid()
#             # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#             #
#             # color = 'tab:blue'
#             # ax2.set_ylabel((f'{label} kg/s'),
#             #                color=color)  # we already handled the x-label with ax1
#             # ax2.scatter(N1, scaling_F(re_dp, re, polynomials[x_iter], polynomials[x_iter + 1]), color=color)
#             # ax2.tick_params(axis='y', labelcolor=color)
#
#             # fig.suptitle(title + " " + flight_cond)
#             # fig.tight_layout()  # otherwise the right y-label is slightly clipped
#             # plt.savefig(directory_path + "/" + title + " " + flight_cond + ".jpg")
#             # plt.show()
#
#
#     if i == 0:
#         print("take off start")
#         X = X_takeoff
#     elif i == 1:
#         print("climb start")
#         X = X_climb
#     else:
#         print("cruise start")
#         X = X_cruise
#
#     LPCc_X = X[:6]
#     LPCd_X = X[6:12]
#     HPC_X = X[12:18]
#     HPT_X = X[18:20]
#     LPT_X = X[20:22]
#
#     tm_plot(y_array=[PR_HPC, ETA_HPC], tm_type="HPC", x_iters=[0, 4], polynomials=HPC_X, re_dp=Re3_DP, re=Re3)
#     tm_plot(y_array=[PRc_Fan, ETAc_Fan], tm_type="LPCc", x_iters=[0, 4], polynomials=LPCc_X, re_dp=Re25_DP, re=Re25)
#     tm_plot(y_array=[PRd_Fan, ETAd_Fan], tm_type="LPCd", x_iters=[0, 4], polynomials=LPCd_X, re_dp=Re19_DP, re=Re19)
#     tm_plot(y_array=[PR_HPT, ETA_HPT], tm_type="HPT", x_iters=[0], polynomials=HPT_X, re_dp=Re49_DP, re=Re49)
#     tm_plot(y_array=[PR_LPT, ETA_LPT], tm_type="LPT", x_iters=[0], polynomials=LPT_X, re_dp=Re5_DP, re=Re5)
#     Mass_flow_plots()

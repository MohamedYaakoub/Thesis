import pickle
import numpy as np
import timeit
import sys
from GSP_helper import cleanup, runGsp
from matplotlib import pyplot as plt
from map_functions import reset_maps
from WriteCMapGSP import read_mapC, write_mapC
from WriteTMapGSP import read_mapT, write_mapT
from map_functions import reset_maps, plot_maps

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesignGEnx_Valid_All parameters.mxl"  # "GEnx-1B_V3_test2.mxl"  #

GEnx_OD, GEnx_OD_true, N1cCEOD = pickle.load(open("CEOD_GEnx/CEOD_set_Valid.p", "rb"))

# print(GEnx_OD[1].shape)  # inputDat #this is used for y_sim
# print(GEnx_OD_true[2].shape)  # TrueVal of CEOD ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2"]

inputs_list = ["N1", "P0", "T0", "Mach", "HP"]
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2",
               "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19",
               "OPR", "PRc_Fan", "PRd_Fan", "PR_HPC", "PR_HPT", "PR_LPT",
               "ETAc_Fan", "ETAd_Fan", "Eta_HPC", "Eta_HPT", "Eta_LPT",
               "W14", "W25", "W4", "Wc25",
               "Betamap_HPC", "Nc_HPC"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))

GEnx_OD = np.array([item for sublist in GEnx_OD for item in sublist])
GEnx_OD_true = np.array([item for sublist in GEnx_OD_true for item in sublist])

from my_modified_functions import gspdll


def run_simulation(input):
    gspdll.InitializeModel()
    y_sim = np.array(runGsp(gspdll, input, output_list))

    # y_sim = y_sim[:, :6]  # ignore effs for now
    Reynolds = y_sim[:, 6:14]
    PRs = y_sim[:, 14:20]
    ETAs = y_sim[:, 20:25]
    Ws = y_sim[:, 25:29]
    Map = y_sim[:, 29:]
    return PRs, ETAs, Ws, Map


def scale_maps_reynolds(typef, ReDP, ReOD, file_name, poly_param):
    X = poly_param

    if typef == 'C':
        MdotC, EtaC, PRC, surge_mC, surge_pC, NC = pickle.load(open("Constants/" + file_name + "pick.p", "rb"))
        fpr = 1
        fm = 1.1
        fe = 1

        # fsurge_pr = 1
        # fsurge_m  = 2

        surge_pr_arr = surge_pC[1:] * fpr
        surge_mC_arr = surge_mC[1:] * fm

        # fsurge_m  = np.insert(fsurge_m, 0, 1)
        # fsurge_pr = np.insert(fsurge_pr, 0, 1)

        write_mapC(file_name, file_name, np.clip(MdotC * fm, 0.05, 2000), np.clip(EtaC * fe, 0.10101, 0.99),
                   np.clip(PRC * fpr, 0.05, 100), np.clip(np.insert(surge_mC_arr, 0, surge_mC[0]), 0.05, 2000),
                   np.clip(np.insert(surge_pr_arr, 0, surge_pC[0]), 0.05, 100), NC)
    else:
        PRmin, PRmax, MdotT, EtaT, NPrT, NT, BT = pickle.load(open("Constants/" + file_name + "pick.p", "rb"))
        # fpr = scaling_F(Ndp / 100, NPrT[1:], X[0], X[1])
        # fm  = scaling_F(Ndp / 100, NT, X[2], X[3])
        fe = 1
        # fpr = np.insert(fpr, 0, 1)

        write_mapT(file_name, file_name, np.clip(PRmin * 1, 0, 100), np.clip(PRmax * 1, 0, 100), np.clip(MdotT * 1,
                                                                                                         0.05, 2000),
                   np.clip(EtaT * fe, 0.10101, 0.99), NT, BT)


reset_maps()

GEnx_OD = GEnx_OD[:20]

PRs1, ETAs1, Ws1, Map1 = run_simulation(GEnx_OD)

# scale_maps_reynolds("C", 0, 0, "1_LPC_core", 0)
# scale_maps_reynolds("C", 0, 0, "2_LPC_bypass", 0)
# scale_maps_reynolds("C", 0, 0, "3_HPC", 0)
# scale_maps_reynolds("T", 0, 0, "4_HPT", 0)
# scale_maps_reynolds("T", 0, 0, "5_LPT", 0)
# scale_maps_reynolds("C", None, None, "3_HPC", None)

PRs2, ETAs2, Ws2, Map2 = run_simulation(GEnx_OD)

plt.scatter(GEnx_OD[:, 0], PRs1[:, 3], c='black')
plt.scatter(GEnx_OD[:, 0], PRs2[:, 3], c='blue')
plt.ylabel('PR HPC')
plt.xlabel('N1')
plt.tight_layout()
plt.show()

plt.scatter(GEnx_OD[:, 0], PRs1[:, 0], c='black')
plt.scatter(GEnx_OD[:, 0], PRs2[:, 0], c='blue')
plt.ylabel('OPR')
plt.xlabel('N1')
plt.tight_layout()
plt.show()

plt.scatter(GEnx_OD[:, 0], ETAs1[:, 2], c='black')
plt.scatter(GEnx_OD[:, 0], ETAs2[:, 2], c='blue')
plt.ylabel('Eta HPC')
plt.xlabel('N1')
plt.tight_layout()
plt.show()

plt.scatter(GEnx_OD[:, 0], Ws1[:, 3], c='black')
plt.scatter(GEnx_OD[:, 0], Ws2[:, 3], c='blue')
plt.ylabel('WC25')
plt.xlabel('N1')
plt.tight_layout()
plt.show()

plt.scatter(GEnx_OD[:, 0], Map1[:, 0], c='black')
plt.scatter(GEnx_OD[:, 0], Map2[:, 0], c='blue')
plt.ylabel('Beta_HPC')
plt.xlabel('N1')
plt.tight_layout()
plt.show()


plt.scatter(GEnx_OD[:, 0], Map1[:, 1], c='black')
plt.scatter(GEnx_OD[:, 0], Map2[:, 1], c='blue')
plt.ylabel('Nc_hpc')
plt.xlabel('N1')
plt.tight_layout()
plt.show()
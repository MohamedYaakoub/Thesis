import pickle
import numpy as np
from GSP_helper import cleanup, runGsp
import os

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesignGEnx_Valid_All parameters - no OD.mxl"  # "GEnx-1B_V3_test2.mxl"  #

inputs_list = ["N1", "P0", "T0", "Mach", "HP"]
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2",
               "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19", "Re_6", "Re_9",
               "OPR", "PRc_Fan", "PRd_Fan", "PR_HPC", "PR_HPT", "PR_LPT", "PR_d", "PR_nozzle", "PR_bypass",
               "ETAc_Fan", "ETAd_Fan", "Eta_HPC", "Eta_HPT", "Eta_LPT",
               "W14", "W25", "W4", "Wc2", "Wcc_fan", "Wcd_Fan", "Wc25", "Wc4", "Wc49",
               "TRc_fan", "TRd_fan", "TR_HPC", "TR_HPT", "TR_LPT", "TR_d", "TR_nozzle", "TR_bypass",
               "FN", "FG_nozzle", "FG_bypass",
               "Tt2", "Tt14", "Tt25", "Tt3", "Tt4", "Tt49", "Tt5",
               "Pt2", "Pt14", "Pt25", "Pt3", "Pt4", "Pt49", "Pt5",
               "Nc_fan", 'Nc_HPC', 'Nc_HPT', 'Nc_LPT']
# "Betac_fan", 'Betad_fan', 'Beta_HPC', 'Beta_HPT', 'Beta_LPT',
# "M25", "M14", 'M3', "M49", "M5"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))

from my_modified_functions import gspdll


def get_Reynolds(ceod_file):
    # GEnx_OD, GEnx_OD_true, N, alt_time = pickle.load(open(f"CEOD_GEnx/same_engine_flights/{ceod_file}", "rb"))
    GEnx_OD, GEnx_OD_true, N, alt_time = pickle.load(open(ceod_file, "rb"))
    ceod_file = 'Clusters_v1.p'
    print(ceod_file)

    def simulate(inputDat):
        y_sim = np.array(runGsp(gspdll, inputDat, output_list))
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

        return y_sim_valid, Reynolds, PRs, ETAs, Ws, TR, FN, Tt, Pt, Nc

    All_validation_params = []
    All_Reynolds = []
    ALL_PRs = []
    All_ETAs = []
    All_Ws = []
    All_TR = []
    All_Fn = []
    All_Tt = []
    All_Pt = []
    All_Nc = []

    gspdll.InitializeModel()
    for inputDat in GEnx_OD:
        print('Loop initiated')
        print(inputDat.shape)
        y_sim_valid, Reynolds, PRs, ETAs, Ws, TR, FN, Tt, Pt, Nc = simulate(inputDat)

        All_validation_params.append(y_sim_valid)
        All_Reynolds.append(Reynolds)
        ALL_PRs.append(PRs)
        All_ETAs.append(ETAs)
        All_Ws.append(Ws)
        All_TR.append(TR)
        All_Fn.append(FN)
        All_Tt.append(Tt)
        All_Pt.append(Pt)
        All_Nc.append(Nc)
        print(FN)
        print(y_sim_valid.shape)


    # pickle.dump([GEnx_OD, GEnx_OD_true, N, alt_time,
    #              All_Reynolds, All_validation_params, ALL_PRs, All_ETAs, All_Ws, All_TR, All_Fn, All_Tt, All_Pt,
    #              All_Nc], open("Reynolds_pickle/Reynolds_" + ceod_file, "wb"))

    #for clustering
    pickle.dump([GEnx_OD, GEnx_OD_true,
                 All_Reynolds, All_validation_params, ALL_PRs, All_ETAs, All_Ws, All_TR, All_Fn, All_Tt, All_Pt,
                 All_Nc], open("Reynolds_pickle/Reynolds_" + ceod_file, "wb"))
    print(ceod_file, 'done')


if __name__ == '__main__':

    # file_name = "CEOD_one_flight_sampled_no_Reynolds.p"
    # file_name = "CEOD_data_mohamed_2019_feb_1-9_2.p"
    directory = "CEOD_GEnx/same_engine_flights/"
    get_Reynolds('Clusters/Clusters_v1.p')
    # for file in os.listdir(directory)[2:]:
    #     try:
    #         get_Reynolds(file)
    #     except:
    #         pass

    # file_name = "CEOD_200408-203904-KLM168____-KATLEHAM-KL_PH-BHA-2-956609-W010FFD.P"
    # file_name = "CEOD_160724-193429-KLM891____-EHAMZUUU-KL_PH-BHA-2-956609-W007FFD.p"
    # get_Reynolds(file_name)

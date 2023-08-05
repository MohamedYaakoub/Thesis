from map_functions import reset_maps, plot_maps
import pickle
import numpy as np
from WriteCMapGSP import read_mapC, write_mapC
from WriteTMapGSP import read_mapT, write_mapT

# reset_maps()
#
# plot_maps('C', "1_LPC_core")
# plot_maps('C', "2_LPC_bypass")

# plot_maps('T', "4_HPT")
# plot_maps('T', "5_LPT")

def scale_maps_reynolds(typef, ReDP, ReOD, file_name, poly_param):
    X = poly_param

    if typef == 'C':
        MdotC, EtaC, PRC, surge_mC, surge_pC, NC = pickle.load(open("Constants/" + file_name + "pick.p", "rb"))
        fpr = 1
        fm  = 1.1
        fe  = 1


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
        fe  = 1
        # fpr = np.insert(fpr, 0, 1)

        write_mapT(file_name, file_name, np.clip(PRmin * 1, 0, 100), np.clip(PRmax * 1, 0, 100), np.clip(MdotT * 1,
                                                                 0.05, 2000), np.clip(EtaT * fe, 0.10101, 0.99), NT, BT)

reset_maps()
plot_maps('C', "3_HPC")
# scale_maps_reynolds("C", None, None, "3_HPC", None)
# scale_maps_reynolds("C", 0, 0, "1_LPC_core", 0)
scale_maps_reynolds("C", 0, 0, "2_LPC_bypass", 0)
plot_maps('C', "2_LPC_bypass")
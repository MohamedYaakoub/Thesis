import pickle
import numpy as np
from matplotlib import pyplot as plt
from WriteCMapGSP import read_mapC, write_mapC
from WriteTMapGSP import read_mapT, write_mapT


def reset_maps():
    for mapi in ["1_LPC_core", "2_LPC_bypass", "3_HPC", "4_HPT", "5_LPT"]:
        if "PC" in mapi:
            MdotC, EtaC, PRC, surge_mC, surge_pC, NC = read_mapC(mapi, mapi, 0)
            pickle.dump([MdotC, EtaC, PRC, surge_mC, surge_pC, NC], open("Constants/" + mapi + "pick.p", "wb"))
            write_mapC(mapi, mapi, MdotC, EtaC, PRC, surge_mC, surge_pC, NC)
        else:
            PRmin, PRmax, MdotT, EtaT, NPrT, NT, BT = read_mapT(mapi, mapi, 0)
            pickle.dump([PRmin, PRmax, MdotT, EtaT, NPrT, NT, BT], open("Constants/" + mapi + "pick.p", "wb"))
            write_mapT(mapi, mapi, PRmin, PRmax, MdotT, EtaT, NT, BT)


def plot_maps(typef, file_name):
    if typef == 'C':
        # plot the reference and modified maps for the compressors
        # read the reference and modified map
        MdotR, EtaR, PRR, surge_mR, surge_pR, NR = read_mapC(file_name, file_name, 0)
        Mdot, Eta, PR, surge_m, surge_p, N = read_mapC(file_name, file_name, 1)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # plot the reference map
        ax1.scatter(MdotR, PRR,c='grey', marker=".", edgecolors='grey', s=50)
        ax1.plot(surge_mR[1:], surge_pR[1:], c='red')
        # plot the modified map
        ax1.scatter(Mdot, PR, c='k', marker=".", edgecolors='k', s=50)
        ax1.plot(surge_m[1:], surge_p[1:], c='red', linestyle="-.")
        ax1.set_xlabel("Corrected Massflow")
        ax1.set_ylabel("Pressure Ratio")
        ax1.grid()
        ax1.set_title(file_name)
        #
        ax2.scatter(MdotR, EtaR, c='grey', marker=".", edgecolors='grey', s=50)  # plot the reference map
        ax2.scatter(Mdot, Eta, c='k', marker=".", edgecolors='k', s=50)  # plot the generated maps
        ax2.set_xlabel("Corrected Massflow")
        ax2.set_ylabel("Efficiency")
        ax2.grid()
        ax2.set_title(file_name)
        plt.tight_layout()
        plt.show()

    else:
        # plot the reference and modified maps for the turbines
        # read the reference and modified map
        PRminR, PRmaxR, MdotTR, EtaTR, NPrTR, NTR, B = read_mapT(file_name, file_name, 0)
        PRmin, PRmax, MdotT, EtaT, NPrT, NT, B = read_mapT(file_name, file_name, 1)
        # extraxt the iso speed lines
        no_betalines = len(B)
        listsER = np.array([EtaTR[x:x + no_betalines] for x in range(0, len(EtaTR), no_betalines)], dtype=object)
        listsMR = np.array([MdotTR[x:x + no_betalines] for x in range(0, len(MdotTR), no_betalines)], dtype=object)
        listsE  = np.array([EtaT[x:x + no_betalines] for x in range(0, len(EtaTR), no_betalines)], dtype=object)
        listsM  = np.array([MdotT[x:x + no_betalines] for x in range(0, len(MdotTR), no_betalines)], dtype=object)

        listsPR_R = np.zeros(listsER.shape)
        listsPR   = np.zeros(listsE.shape)

        for i in range(len(PRmax) - 1):
            p_ref  = np.polyfit([0, 1], [PRminR[i + 1], PRmaxR[i + 1]], 1)
            p      = np.polyfit([0, 1], [PRmin[i + 1], PRmax[i + 1]], 1)
            PR_ref = np.polyval(p_ref, B)
            PR     = np.polyval(p, B)
            listsPR_R[i, 0:no_betalines] = PR_ref
            listsPR[i, 0:no_betalines]   = PR

        # plotting
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for i in range(len(listsER)):
            ax1.plot(listsPR_R[i], listsER[i], linestyle="-.", c='grey')  # plot the reference map
            ax1.plot(listsPR[i], listsE[i], c='k')  # plot the modified map

            ax2.plot(listsPR_R[i], listsMR[i], linestyle="-.", c='grey')  # plot the reference map
            ax2.plot(listsPR[i], listsM[i], c='k')  # plot the modified map

        ax1.set_xlabel("Pressure Ratio")
        ax1.set_ylabel("Efficiency")
        ax1.grid()
        ax1.set_title(file_name)
        ax2.set_xlabel("Pressure Ratio")
        ax2.set_ylabel("Mass flow")
        ax2.grid()
        ax2.set_title(file_name)
        plt.tight_layout()
        plt.show()

def scaling_F(ReDP, ReOD, a, b):
    """
    Scaling function is a second degree polynomial
    :param ReDP: design spool speed
    :param ReOD: off-design spool speed
    :return: function value
    """
    return np.array(1 + a * ((ReOD - ReDP) / ReDP) + b * ((ReOD - ReDP) / ReDP) ** 2)


# _, All_Reynolds = pickle.load(open("Constants/Reynolds_set_Valid.p", "rb"))
# All_Reynolds = np.array([item for sublist in All_Reynolds for item in sublist])
# Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19 = All_Reynolds.T

# def plot_SF(typef, ReOD_arr, ReDP, file_name, poly_param):
#     X = poly_param
#     Ndp = p.inputs['Np1'] if spool == 1 else p.inputs['Np2']
#     if typef == 'C':
#         plt.plot(
#             scaling_F(Ndp / 100, np.linspace(0.3, np.max(inputDat[:, 0])/100, 50), X[0], X[1]), c='r',
#             label="P")
#         plt.plot(
#             scaling_F(Ndp / 100, np.linspace(0.3, np.max(inputDat[:, 0])/100, 50), X[2], X[3]), c='g',
#             label="M")
#         plt.plot(
#             scaling_F(Ndp / 100, np.linspace(0.3, np.max(inputDat[:, 0])/100, 50), X[4], X[5]), c='b',
#             label="E")
#         plt.legend()
#         plt.title(file_name)
#         plt.show()
#     else:
#         plt.plot(
#             scaling_F(Ndp / 100, np.linspace(0.3, np.max(inputDat[:, 0])/100, 50), X[0], X[1]), c='r',
#             label="E")
#         plt.legend()
#         plt.title(file_name)
#         plt.show()
#
if __name__ == '__main__':
    reset_maps()

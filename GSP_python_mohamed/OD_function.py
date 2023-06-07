"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 5/24/2022 1:28 PM  
"""
import numpy as np
import subprocess
import os
import sys
from WriteCMapGSP import read_mapC, write_mapC
from WriteTMapGSP import read_mapT, write_mapT

sys.path.insert(1, "C:/Users/mohsy/University/KLM/Thesis/My thesis/Parallel GSP/Shared "
                   "Folder/GSP")
from GSP_helper import loadModel, runGsp
import pickle
from parameters import params, params2, CF6_OD, CF6_OD_true, GEnx_OD, GEnx_OD_true
from threading import Thread
import matplotlib.pyplot as plt

# load the following from the file where the objective is called:
# input list : the list specifying the input params
# output list: the list specifying the output params
# GSPfileName:  . . . . .
# Engine     :  0 for the CF6 and 1 for the GEnx
inputs_list, output_list, GSPfileName, Engine = pickle.load(open("io.p", "rb"))

gspdll = loadModel(Engine, GSPfileName)
# print(Engine, GSPfileName)

p = params() if Engine == 0 else params2()      # load the data for the engines
inputDat = CF6_OD if Engine == 0 else GEnx_OD   # The off design control parameters

def reset_maps():
    for mapi in ["1_LPC_core", "2_LPC_bypass", "3_HPC", "4_HPT", "5_LPT"]:
        if "PC" in mapi:
            MdotC, EtaC, PRC, surge_mC, surge_pC, NC = read_mapC(mapi, mapi, 0)
            pickle.dump([MdotC, EtaC, PRC, surge_mC, surge_pC, NC], open(mapi + "pick.p", "wb"))
            write_mapC(mapi, mapi, MdotC, EtaC, PRC, surge_mC, surge_pC, NC)
        else:
            PRmin, PRmax, MdotT, EtaT, NPrT, NT, BT = read_mapT(mapi, mapi, 0)
            pickle.dump([PRmin, PRmax, MdotT, EtaT, NPrT, NT, BT], open(mapi + "pick.p", "wb"))
            write_mapT(mapi, mapi, PRmin, PRmax, MdotT, EtaT, NT, BT)

# %%
def scaling_F(Ndp, Nod, a, b):
    """
    Scaling function is a second degree polynomial
    :param Ndp: design spool speed
    :param Nod: off-design spool speed
    :return: function value
    """
    return np.array(1 + a * ((Nod - Ndp) / Ndp) + b * ((Nod - Ndp) / Ndp) ** 2)



def scale_maps(typef, spool, file_name, poly_param):
    X = poly_param
    Ndp = p.inputs['Np1'] if spool == 1 else p.inputs['Np2']

    if typef == 'C':
        MdotC, EtaC, PRC, surge_mC, surge_pC, NC = pickle.load(open(file_name + "pick.p", "rb"))
        fpr = scaling_F(Ndp / 100, NC, X[0], X[1])
        fm  = scaling_F(Ndp / 100, NC, X[2], X[3])
        fe  = scaling_F(Ndp / 100, NC, X[4], X[5])

        surgeN = np.unique(NC)

        fsurge_pr = scaling_F(Ndp / 100, surgeN, X[0], X[1])
        fsurge_m  = scaling_F(Ndp / 100, surgeN, X[2], X[3])
        fsurge_m  = np.insert(fsurge_m, 0, 1)
        fsurge_pr = np.insert(fsurge_pr, 0, 1)

        write_mapC(file_name, file_name, np.clip(MdotC * fm, 0.05, 2000), np.clip(EtaC * fe, 0.10101, 0.99),
                   np.clip(PRC * fpr, 0.05, 100), np.clip(surge_mC * fsurge_m, 0.05, 2000),
                   np.clip(surge_pC * fsurge_pr, 0.05, 100), NC)
    else:
        PRmin, PRmax, MdotT, EtaT, NPrT, NT, BT = pickle.load(open(file_name + "pick.p", "rb"))
        # fpr = scaling_F(Ndp / 100, NPrT[1:], X[0], X[1])
        # fm  = scaling_F(Ndp / 100, NT, X[2], X[3])
        fe  = scaling_F(Ndp / 100, NT, X[0], X[1])
        # fpr = np.insert(fpr, 0, 1)

        write_mapT(file_name, file_name, np.clip(PRmin * 1, 0, 100), np.clip(PRmax * 1, 0, 100), np.clip(MdotT * 1,
                                                                 0.05, 2000), np.clip(EtaT * fe, 0.10101, 0.99), NT, BT)

def objFOD(X):
    gspdll.InitializeModel()

    scale_maps("C", 1, "1_LPC_core", X[:6])
    scale_maps("C", 1, "2_LPC_bypass", X[6:12])
    scale_maps("C", 2, "3_HPC", X[12:18])
    scale_maps("T", 2, "4_HPT", X[18:20])
    scale_maps("T", 1, "5_LPT", X[20:22])

    y_true   = CF6_OD_true if Engine == 0 else GEnx_OD_true

    y_sim = np.array(runGsp(gspdll, inputDat, output_list))

    weights = np.ones(len(output_list))

    # add weights if necessary
    # weights[output_list.index("FN")]   = weights[output_list.index("FN")]*10
    # weights[output_list.index("TT25")] = weights[output_list.index("TT25")]*5
    # weights[output_list.index("TT3")]  = weights[output_list.index("TT3")]*5

    Rms = np.sqrt(np.mean(np.mean(((y_true - y_sim) / (y_true + 0.000001)) ** 2, axis=0)*weights))

    return Rms

def objScale(X):
    scale_maps("C", 1, "1_LPC_core", X[:6])
    scale_maps("C", 1, "2_LPC_bypass", X[6:12])
    scale_maps("C", 2, "3_HPC", X[12:18])
    scale_maps("T", 2, "4_HPT", X[18:20])
    scale_maps("T", 1, "5_LPT", X[20:22])


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

def plot_SF(spool, typef, file_name, poly_param):
    X = poly_param
    Ndp = p.inputs['Np1'] if spool == 1 else p.inputs['Np2']
    if typef == 'C':
        plt.plot(
            scaling_F(Ndp / 100, np.linspace(0.3, np.max(inputDat[:, 0])/100, 50), X[0], X[1]), c='r',
            label="P")
        plt.plot(
            scaling_F(Ndp / 100, np.linspace(0.3, np.max(inputDat[:, 0])/100, 50), X[2], X[3]), c='g',
            label="M")
        plt.plot(
            scaling_F(Ndp / 100, np.linspace(0.3, np.max(inputDat[:, 0])/100, 50), X[4], X[5]), c='b',
            label="E")
        plt.legend()
        plt.title(file_name)
        plt.show()
    else:
        plt.plot(
            scaling_F(Ndp / 100, np.linspace(0.3, np.max(inputDat[:, 0])/100, 50), X[0], X[1]), c='r',
            label="E")
        plt.legend()
        plt.title(file_name)
        plt.show()


if __name__ == '__main__':
    y_true = CF6_OD_true if Engine == 0 else GEnx_OD_true

    y_sim = np.array(runGsp(gspdll, inputDat, output_list))

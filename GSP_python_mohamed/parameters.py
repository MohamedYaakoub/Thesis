"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 3/14/2022 3:42 PM  
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

path = os.getcwd()
Foldername = "DataDocs"  # data folder name
#%% The function to apply corrections

def correction(N, W, Pa, Ta, R=False):
    # shaft speed or massflow correction
    # sea level conditions
    Pref = 1.01325
    Tref = 288.15

    delta = Pa / Pref
    theta = Ta / Tref

    if not R:
        Wc = W * np.sqrt(theta) / delta
        Nc = N / np.sqrt(theta)
        return Wc, Nc
    else:
        Wuc = W / (np.sqrt(theta) / delta)
        Nuc = N * np.sqrt(theta)
        return Wuc, Nuc


# %% correlation report data
# For the B5F KLM phase 1, cycle 4
# data from back to back testing is used instead of the correlation report due to the incorrect (or unknown) value
# for the T5 given as the "T5SD". From the data it was observed that this parameter was almost equal to the EGT.
# additionally, it was noticed that the P54 was equal to the P49.

def loadDataCF6():
    # ====================
    # Load data from excel
    # ====================

    DataDocs_filepath = 'C:/Users/mohsy/University/KLM/Thesis/My thesis/Correlation Reports/DataDocs'
    filename   = "all_runs"

    # load data
    correlationData_excel = pd.read_csv(DataDocs_filepath + "\\" + filename + ".csv", skiprows=0,
                                        index_col=False, header=0)

    return correlationData_excel

datCF6C2   = loadDataCF6()

datCF6C2   = datCF6C2[datCF6C2["run"] == "210903_13_40_PCL_4"]

datCF6C2dp = datCF6C2[datCF6C2["reading"] == "b5f_to"].iloc[1]

""" The following conversions are carried out below
# lbf     to kN      for thrust
# pph     to kg/s    for the fuel flow
# pps     to kg/s    for the mass flow
# Psi     to bar     for the pressure
# Deg C   to Kelvin  for the temperature
"""

Fgdp    = datCF6C2dp.loc["fn_net_obs-lbs"] * 0.00444822162
N2dp    = datCF6C2dp.loc["n2_observed-rpm"]
N2pdp   = datCF6C2dp.loc["n2_obs-%"]
N1dp    = datCF6C2dp.loc["n1_observed-rpm"]
N1pdp   = datCF6C2dp.loc["n1_obs-%"]
T2dp    = datCF6C2dp.loc["t2_temp_avg-deg_c"] + 273.15
rhum_dp = datCF6C2dp.loc["humidity-%"]
Pt2dp   = datCF6C2dp.loc["pt2__cell_forward-psia"]  * 0.0689475729
T49dp   = datCF6C2dp.loc["egt_observed_345-deg_c"]  + 273.15
T3dp    = datCF6C2dp.loc["t3-deg_c"]  + 273.15
T25dp   = datCF6C2dp.loc["t25-deg_c"]  + 273.15
Wfdp    = datCF6C2dp.loc["wf_observed-pph"] * 0.0001259979
Ps25dp  = datCF6C2dp.loc["ps25-psia"]  * 0.0689475729
Pt25dp  = datCF6C2dp.loc["pt25-psia"]  * 0.0689475729
PS3dp   = datCF6C2dp.loc["ps3-psia"]  * 0.0689475729
Pt49dp  = datCF6C2dp.loc["pt49-psia"]  * 0.0689475729
T5dp    = datCF6C2dp.loc["t5-deg_c"]  + 273.15
Ps14dp  = datCF6C2dp.loc["ps14-psia"]  * 0.0689475729
W2dp    = datCF6C2dp.loc["wa2k-lb_sec"]  * 0.45359237
W2dp, _ = correction(1, W2dp, Pt2dp, T2dp, True)

true_val_DP_CF6 = np.array([T25dp, Ps14dp, T3dp, PS3dp, Pt49dp, T49dp, T5dp, Fgdp])

PRFanc  = Pt25dp/Pt2dp

# margins for an indication of the engine health
# Thrust margin is in kN
# EGT margin is HD and in deg C
Fn_mar  = datCF6C2dp.loc["fnmarfn_margin-lbs"] * 0.00444822162
EGT_mar = datCF6C2dp.loc["egtmaregt_margin_hd-deg_c"]
N2_mar  = datCF6C2dp.loc["n2marn2_margin_hd-rpm"]

class params:
    def __init__(self):
        """Parameters for the CF6-80"""
        self.inputs = {
            # Inlet Module
            'dot_w': W2dp,
            'in_Aout ': 4.3030,
            # Fan Module
            'fan_A_C': 0.635099952,
            'fan_A_Bp': 2.892394607,
             'PRFcore': PRFanc,
            'Np1': N1pdp,
            'N1': N1dp,
            # HPC Module
            'N2': N2dp,
            'Np2': N2pdp,
            'HPC_A': 0.067675228,  # for T3
            # Combustor
            'Wf': Wfdp,
            'combM': 0,
            'comb_eta': 0.995,
            'comb_loss': 0.04,
            # bypass nozzle
            'CV_bp': 1,
            'CX_bp': 1,
            'CD_bp': 1,
            # core nozzle
            'CV_c': 1,
            'CX_c': 1,
            'CD_c': 1,
            # Ambient (old data)
            'hum': rhum_dp,
            'Ma': 0,
            'Ta': T2dp,
            'Pa': Pt2dp,

        }
        self.stagesHPC = 14
        self.stagesHPT = 2
        self.stagesLPT = 5
        self.stagesB   = 5
        # Nozzle areas
        self.Abp = 1.77
        self.Ac = 0.615
        # radii
        self.RmfanC   = 0.605000455
        self.RmfanBp  = 0.9347685
        self.RmHPC    = 0.274331807
        self.RmHPTin  = 0.392183952
        self.RmHPTout = 0.393330517
        self.RmLPTin  = 0.400582637
        self.RmLPTout = 0.511059681
        self.RmLPT    = 0.473
        # HPC inlet area
        self.HPCinlA = 0.339612024
        self.HPCoutA = 0.067675228
        self.OPR = 31.1  # from janes https://customer.janes.com/Janes/Display/JAE_0731-JAE_ for the B6F

pcf6 = params()

# %% GEnx data
# Rating GEnX - 1B74_75P2
# The EGT margin is: 10.93361 degC
""" The following conversions are carried out below
# lb      to kN      for thrust
# pph     to kg/s    for the fuel flow
# Psi     to bar     for the pressure
# Deg C   to Kelvin  for the temperature
"""


def massflow(Ps, Pt, Tt, A=5.58644):
    # area is for the GEnx, P is in bar and Tt in K
    # Pt = Ps + 0.5*rho*v**2
    R = 287
    Cp = 1000
    Pdyn = (Pt - Ps) * 1e5
    rho = (Ps * 1e5 / R + Pdyn / Cp) / Tt
    V = np.sqrt(Pdyn * 2 / rho)
    W = A * rho * V
    print(V, rho, W)
    return W


def loadDataGEnx():
    # ====================
    # Load data from excel
    # ====================

    filenameRaw = "GEnx baseline data"
    filenameC = "GEnx baseline data C"
    filename = "genx_correlation_pc run5"

    # Load correlation report data
    # correlationDataRaw_excel = pd.read_excel(path + "\\" + Foldername + "\\" + filenameRaw + ".xlsx",
    #                                         skiprows=1, index_col=0)

    DataDocs_filepath = 'C:/Users/mohsy/University/KLM/Thesis/My thesis/Correlation Reports/DataDocs'
    correlationDataRaw_excel = pd.read_excel(DataDocs_filepath + "\\" + filenameRaw + ".xlsx",
                                            skiprows=1, index_col=0)

    # `
    # correlationDataRawC_excel = pd.read_excel(path + "\\" + Foldername + "\\" + filenameC + ".xlsx",
    #                                           skiprows=0, index_col=0)`

    correlationDataRawC_excel = pd.read_excel(DataDocs_filepath + "\\" + filenameC + ".xlsx",
                                              skiprows=0, index_col=0)

    # correlationData_excel = pd.read_excel(path + "\\" + Foldername + "\\" + filename + ".xls ", skiprows=[1, 2],
    #                                       index_col=0, sheet_name=1)

    correlationData_excel = pd.read_excel(DataDocs_filepath + "\\" + filename + ".xls ", skiprows=[1, 2],
                                          index_col=0, sheet_name=1)

    correlationData_excel = correlationData_excel.loc[:, ~correlationData_excel.columns.str.contains('^Unnamed')]

    return correlationDataRaw_excel, correlationDataRawC_excel, correlationData_excel


datGEnxR, datGEnxC, datGEnx = loadDataGEnx()

DPgenxAll = datGEnx['PC_TO_1B7475_2']

T12dpgx = DPgenxAll.loc['T12_SEL_ChA'] + 273.15
T2dpgx = DPgenxAll.loc['TT2'] + 273.15
TT3dpgx = DPgenxAll.loc['T3_SEL_ChA'] + 273.15
TT25dpgx = DPgenxAll.loc['T25_SEL_ChA'] + 273.15
TT49dpgx = DPgenxAll.loc['EGT_SEL_ChA'] + 273.15
Pt0dpgx = DPgenxAll.loc['P0_SEL_ChA'] * 0.0689475729
Ps10dpgx = DPgenxAll.loc['PS10W_Avg'] * 0.0689475729
Pt10dpgx = DPgenxAll.loc['PT10_Avg'] * 0.0689475729
Pt2dpgx = DPgenxAll.loc['PT2_SEL_ChA'] * 0.0689475729
Ps25dpgx = DPgenxAll.loc['PS25_ChA'] * 0.0689475729
Ps3dpgx = DPgenxAll.loc['PS3_SEL_ChA'] * 0.0689475729
N1dpgx = DPgenxAll.loc['N1']
N1KSDdpgx = DPgenxAll.loc['N1KSD']  # to read from table
N2dpgx = DPgenxAll.loc['N2']
N1pdpgx = DPgenxAll.loc['N1_SEL_ChA_%']
N2pdpgx = DPgenxAll.loc['N2_SEL_ChA_%']
Fndpgx = DPgenxAll.loc['FNnet'] * 0.004448
Rhumdpgx = DPgenxAll.loc['RHUM']
Wfdpgx = DPgenxAll.loc['WF'] * 0.0001259979
W2dpgx = np.interp(N1KSDdpgx, np.flip(datGEnxC["ATC.N1"]), np.flip(datGEnxC["ATC.W2A"])) * 0.45359237
W2dpgx, _ = correction(1, W2dpgx, Pt10dpgx, T2dpgx, True)

true_val_DP_GEnx = np.array([TT25dpgx, TT3dpgx, Ps3dpgx, TT49dpgx, Fndpgx])

g     = 1.4
Ts    = T2dpgx/(Pt10dpgx/Ps10dpgx)**((g-1)/g)
Mach  = np.sqrt((T2dpgx/Ts - 1)/(0.5*(g - 1)))
rho   = 10**5*Ps10dpgx/(287*Ts)
v     = Mach * np.sqrt(g*287*Ts)

Mdot  = rho*v*5.5864

# margins for an indication of the engine health
# Thrust margin is %
# EGT margin is HD and in deg C
Fn_margx  = DPgenxAll.loc["FNK_MARG%"]
EGT_margx = DPgenxAll.loc["EGTHD_MARG"]
N2_margx  = DPgenxAll.loc["N2HD_MARG"]

class params2:
    """Parameters for the GEnx"""

    def __init__(self):
        self.inputs = {
            # Inlet Module
            'dot_w': W2dpgx,
            # Fan Module
            'fan_A_C': 0.610140301,
            'fan_A_Bp': 4.567159077,  # was zero
            'Np1': N1pdpgx,
            'N1': N1dpgx,
            # HPC Module
            'N2': N2dpgx,
            'Np2': N2pdpgx,
            'HPC_A': 0.082344939,  # for T3
            # Combustor
            'Wf': Wfdpgx,
            'combM': 0,
            'comb_eta': 0.995,
            'comb_loss': 0.04,
            # bypass nozzle
            'CV_bp': 1,
            'CX_bp': 1,
            'CD_bp': 1,
            # core nozzle
            'CV_c': 1,
            'CX_c': 1,
            'CD_c': 1,
            # Ambient (old data)
            'hum': Rhumdpgx,
            'Ma': 0,
            'Ta': T2dpgx,
            'Pa': Pt10dpgx,

        }
        self.stagesHPC = 10
        self.stagesHPT = 2
        self.stagesLPT = 7
        self.stagesB   = 5
        # Nozzle areas
        self.Abp = 3.0968
        self.Ac = 0.6698
        # radii
        self.RmfanC   = 0.659190707
        self.RmfanBp  = 1.071908408
        self.RmHPC    = 0.288237949
        self.RmHPTin  = 0.392183952 # 0.368402353
        self.RmHPTout = 0.368712109
        self.RmLPTin  = 0.547234438
        self.RmLPTout = 0.547234438
        self.RmLPT    = 0.649625787
        # HPC inlet area
        self.HPCinlA = 0.358647582
        self.HPCoutA = 0.082344939
        self.OPR = 46.3  # from https://www.geaviation.com/commercial/engines/genx-engine for the -1B74/75 (787-9)


pGEnx = params2()


#%% Off design data for the CF6

datCF6C2 = loadDataCF6()

datCF6C2C   = datCF6C2.copy()

datCF6C21   = datCF6C2[datCF6C2C["run"] == "210903_13_40_PCL_4"]
# datCF6C22   = datCF6C2C[datCF6C2C["run"] == "210902_20_16_PCL_3"]

Fg_list       = datCF6C21["fn_net_obs-lbs"] * 0.00444822162
N1p_list      = datCF6C21["n1_obs-%"]
bln_list      = N1p_list > 80
T3_list       = datCF6C21["t3-deg_c"]  + 273.15
T2_list       = datCF6C21["t2_temp_avg-deg_c"] + 273.15
rhum_list     = datCF6C21["humidity-%"]
Pt2_list      = datCF6C21["pt2__cell_forward-psia"]  * 0.0689475729
N2_list       = datCF6C21["n2_observed-rpm"]
N2p_list      = datCF6C21["n2_obs-%"]
N1p_list      = datCF6C21["n1_obs-%"]
T49_list      = datCF6C21["egt_observed_345-deg_c"]  + 273.15
T25_list      = datCF6C21["t25-deg_c"]  + 273.15
Wf_list       = datCF6C21["wf_observed-pph"] * 0.0001259979
Pt25_list     = datCF6C21["pt25-psia"]  * 0.0689475729
PS3_list      = datCF6C21["ps3-psia"]  * 0.0689475729
Pt49_list     = datCF6C21["pt49-psia"]  * 0.0689475729
T5_list       = datCF6C21["t5-deg_c"]  + 273.15
Ps14_list     = datCF6C21["ps14-psia"]  * 0.0689475729
W2_list       = datCF6C21["wa2k-lb_sec"]  * 0.45359237
W2_list, _    = correction(1, W2_list, Pt2_list, T2_list, True)

Fg_list       = Fg_list[bln_list]
N1p_list      = N1p_list[bln_list]
T2_list       = T2_list[bln_list]
Pt2_list      = Pt2_list[bln_list]
rhum_list     = rhum_list[bln_list]

CF6_OD_true     = np.vstack((N1p_list, T25_list[bln_list], Pt25_list[bln_list], Ps14_list[bln_list], T3_list[bln_list],
                             PS3_list[bln_list], Pt49_list[bln_list], T49_list[bln_list], T5_list[bln_list],
                             Fg_list, W2_list[bln_list], Wf_list[bln_list], N2p_list[bln_list])).T
CF6_OD_true     = CF6_OD_true[CF6_OD_true[:, 0].argsort()]

ind             = []  # find indices of points that are close to each other, these will be deleted
for i, speed in enumerate(CF6_OD_true[:, 0][:-1]):
    if speed*0.995 < CF6_OD_true[:, 0][i+1] < speed*1.005:
        ind.append(i)

CF6_OD_true = np.delete(CF6_OD_true, ind, 0)
CF6_OD_true = np.delete(CF6_OD_true, [-1, -2, -3], 0)  # remove DP and one lower N point
CF6_OD_true = np.flip(CF6_OD_true, axis=0)
# N2p_list    = CF6_OD_true[:, -1]
CF6_OD_true     = CF6_OD_true[:, 1:]
CF6_OD          = np.vstack((N1p_list, Pt2_list, T2_list, rhum_list)).T
CF6_OD          = CF6_OD[CF6_OD[:, 0].argsort()]
CF6_OD          = np.delete(CF6_OD, ind, 0)
CF6_OD          = np.delete(CF6_OD, [-1, -2, -3], 0)  # remove DP
CF6_OD          = np.flip(CF6_OD, axis=0)

# if __name__ == "__main__":
#     plt.figure()
#     plt.scatter(N1p_list, N2p_list[bln_list], c='c', marker="o", edgecolors='b', s=80)
#     plt.grid()
#     plt.xlabel("N1 [%]")
#     plt.ylabel("FN [kN]")
#     # plt.savefig('tessstttyyy.png', dpi=300)
#     plt.show()

# %% for the GEnx below
datGEnx    = datGEnx.drop('Unit', axis=1)  # remove the unit column
#%%
datGEnx    = datGEnx.loc[:, datGEnx.loc['N1_SEL_ChA_%'].gt(40)]  # remove columns with an N1 % lower than 40

N1p_listgx    = datGEnx.loc['N1_SEL_ChA_%']
Fn_listgx     = datGEnx.loc['FNnet'] * 0.004448
T2_listgx     = datGEnx.loc['TT2'] + 273.15
TT3_listgx    = datGEnx.loc['T3_SEL_ChA'] + 273.15
TT25_listgx   = datGEnx.loc['T25_SEL_ChA'] + 273.15
TT49_listgx   = datGEnx.loc['EGT_SEL_ChA'] + 273.15
Pt10_listgx   = datGEnx.loc['PT10_Avg'] * 0.0689475729
Ps3_listgx    = datGEnx.loc['PS3_SEL_ChA'] * 0.0689475729
N1_listgx     = datGEnx.loc['N1']
N1KSD_listgx  = datGEnx.loc['N1KSD']  # to read from table
N2_listgx     = datGEnx.loc['N2']
N2p_listgx    = datGEnx.loc['N2_SEL_ChA_%']
Rhum_listgx   = datGEnx.loc['RHUM']
Wf_listgx     = datGEnx.loc['WF'] * 0.0001259979
W2_listgx     = np.interp(N1KSD_listgx, np.flip(datGEnxC["ATC.N1"]), np.flip(datGEnxC["ATC.W2A"])) * 0.45359237
W2_listgx, _  = correction(1, W2_listgx, Pt10_listgx, T2_listgx, True)

GEnx_OD_true     = np.vstack((N1p_listgx, TT25_listgx, TT3_listgx, Ps3_listgx, TT49_listgx, Fn_listgx, W2_listgx,
                              Wf_listgx, N2p_listgx)).T
GEnx_OD_true     = GEnx_OD_true[GEnx_OD_true[:, 0].argsort()]

ind             = []  # find indices of points that are close to each other, these will be deleted
for i, speed in enumerate(GEnx_OD_true[:, 0][:-1]):
    if speed*0.995 < GEnx_OD_true[:, 0][i+1] < speed*1.005:
        ind.append(i)

GEnx_OD_true     = GEnx_OD_true[:, 1:]
GEnx_OD_true     = np.delete(GEnx_OD_true, ind, 0)
GEnx_OD_true     = np.delete(GEnx_OD_true, [-1, -2, -3], 0)  # remove points close or higher than DP
GEnx_OD_true     = np.flip(GEnx_OD_true, axis=0)
# N2p_listgx       = GEnx_OD_true[:, -1]
# GEnx_OD_true     = GEnx_OD_true[:, :-1]

GEnx_OD          = np.vstack((N1p_listgx, Pt10_listgx, T2_listgx, Rhum_listgx)).T
GEnx_OD          = GEnx_OD[GEnx_OD[:, 0].argsort()]
GEnx_OD          = np.delete(GEnx_OD, ind, 0)
GEnx_OD          = np.delete(GEnx_OD, [-1, -2, -3], 0)  # remove points close or higher than DP
GEnx_OD          = np.flip(GEnx_OD, axis=0)

# if __name__ == "__main__":
#     plt.figure()
#     plt.scatter(N1p_listgx, N2p_listgx, c='orange', marker="o", edgecolors='r', s=80)
#     plt.grid()
#     plt.xlabel("N1 [%]")
#     plt.ylabel("FN [kN]")
#     # plt.savefig('tessstttyyy.png', dpi=300)
#     plt.show()

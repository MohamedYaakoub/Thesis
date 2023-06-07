"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 4/8/2022 4:59 PM  
"""
import numpy as np


def correction(N, W, Pa, Ta, R=False):
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


class params:
    def __init__(self):
        """Parameters for the CF6-80"""
        self.inputs = {
            # Inlet Module
            'dot_w': 824.630,
            'in_Aout ': 4.3030,
            # Fan Module
            'fan_A_C': 0.635099952,
            'fan_A_Bp': 2.892394607,
            'fan_efC': 0.905,
            'fan_efBp': 0.905,  # 0.897
            'BPR': 5.1265,
            'fanCore_pr': 2.5752,
            'fanBp_pr': 1.7265,
            'Np1': 107.47,
            'N1': 3525,
            # HPC Module
            'N2': 10537,
            'Np2': 107.22,
            'HPC_ef': 0.8783,
            'HPC_pr': 13.0115,
            'HPC_A': 0.067675228,  # for T3
            # Combustor
            'Wf': 2.688,
            'combM': 0.25642,
            'comb_eta': 0.995,
            'comb_loss': 0.04,
            # HPT module
            'HPT_ef': 0.8865,
            'HPT_N': 10537,
            'HPT_Np': 107.22,
            'HPT_A': 0.2470,
            # LPT module
            'LPT_ef': 0.9,
            'LPT_N': 3525,
            'LPT_Np': 107.47,
            'LPT_A': 0.9340,
            # bypass nozzle
            'CV_bp': 0.915,
            'CX_bp': 0.9,
            'CD_bp': 0.9,
            # core nozzle
            'CV_c': 0.9,
            'CX_c': 0.9,
            'CD_c': 0.9,
            # Ambient (old data)
            'hum': 0.90957,
            'Ma': 0,
            'Ta': 289.46,
            'Pa': 1.0068,

        }
        self.output_list = ["PT25", "TT25", "PS14", "TT3", "PS3", "TT49",  "PT49", "TT5", "FN"]
        self.stages = 14
        # Nozzle areas
        self.Abp = 1.77
        self.Ac = 0.615
        # radii
        self.RmfanC = 0.605000455
        self.RmfanBp = 0.9347685
        self.RmHPC = 0.274331807
        # HPC inlet area
        self.HPCinlA = 0.339612024
        self.HPCoutA = 0.067675228
        self.OPR = 31.3  # from janes https://customer.janes.com/Janes/Display/JAE_0731-JAE_ for the B6F


class params2:
    """Parameters for the GEnx"""

    def __init__(self):
        self.inputs = {
            # Inlet Module
            'dot_w': 1151.1166,
            'in_Aout ': 0,
            # Fan Module
            'fan_A_C': 0.610140301,
            'fan_A_Bp': 4.567159077,  # was zero
            'fan_efC': 0.9,
            'fan_efBp': 0.92,  # 0.897
            'BPR': 9.0181,
            'fanCore_pr': 2.2296,
            'fanBp_pr': 1.5482,
            'Np1': 96.96,
            'N1': 2482,
            # HPC Module
            'N2': 12491,
            'Np2': 109.79,
            'HPC_ef': 0.872,
            'HPC_pr': 20.7521,
            'HPC_A': 0.082344939,  # for T3
            # Combustor
            'Wf': 2.6708,
            'combM': 0,
            'comb_eta': 0.995,
            'comb_loss': 0.04,
            # HPT module
            'HPT_ef': 0.93,
            'HPT_N': 12651,
            'HPT_Np': 111.2,
            'HPT_A': 0,
            # LPT module
            'LPT_ef': 0.94,
            'LPT_N': 16540,
            'LPT_Np': 100,
            'LPT_A': 0,
            # bypass nozzle
            'CV_bp': 0.915,
            'CX_bp': 0.9,
            'CD_bp': 0.9,
            # core nozzle
            'CV_c': 0.9,
            'CX_c': 0.9,
            'CD_c': 0.9,
            # Ambient (old data)
            'hum': 71.22,  # rel humidity
            'Ma': 0,
            'Ta': 285.59,
            'Pa': 0.9867,

        }
        self.output_list = ["TT25", "TT3", "PS3", "TT49", "FN"]
        self.stages = 10
        # Nozzle areas
        self.Abp = 3.0968
        self.Ac  = 0.6698
        # radii
        self.RmfanC = 0.659190707
        self.RmfanBp = 1.071908408
        self.RmHPC = 0.288237949
        # HPC inlet area
        self.HPCinlA = 0.358647582
        self.HPCoutA = 0.082344939
        self.OPR = 46.3  # from https://www.geaviation.com/commercial/engines/genx-engine for the -1B74/75 (787-9)


# %% correlation report data
# For the BF6 KLM phase 1, cycle 4
N1k_rated = 3471  # from fig 1301 acceptance report

Fg = np.array([56495, 56485, 54965, 52782, 48652, 44837, 40990, 37086])

N1 = np.array([3572.1, 3527.58, 3465.37, 3402.06, 3279.38, 3178.64, 3069.85, 2966.75])

N2 = np.array([10468.4, 10453.5, 10393.5, 10313.5, 10155.4, 10032.2, 9905.08, 9787.65])

T2 = np.array([16.4177, 15.8161, 15.533, 15.4058, 15.5346, 15.422, 15.3746, 15.3191])

rHum = np.array([53.13, 53.2, 56.07, 56.7, 56.63, 56, 56, 56])

Ps0 = np.array([10.6627, 10.7927, 10.9228, 11.0912, 11.4352, 11.7231, 12.0093, 12.2821])

Pt10 = np.array([14.4651, 14.5261, 14.5358, 14.5336, 14.5346, 14.5386, 14.5418, 14.5405])

T49 = np.array([899.912, 893.058, 879.909, 862.336, 826.551, 796.363, 763.846, 733.093])

T3 = np.array([557.619, 552.667, 544.289, 534.846, 516.629, 500.982, 483.808, 467.427])

T25 = np.array([116.622, 112.589, 108.566, 105.113, 99.9695, 95.9084, 91.5919, 87.2117])

Wf = np.array([21873, 21495.7, 20600.9, 19585.6, 17570.3, 16005.8, 14381.5, 12918.9])  # in pph = * 0.0001259979
# np.array([53.9232, 53.16, 50.9733, 48.4314, 43.4369, 39.5446, 35.5318, 31.9052])

Ps25 = np.array([38.7887, 37.8891, 36.6625, 35.6325, 33.855, 32.5928, 31.3775, 30.0875])

Pt56 = np.array([14.6316, 14.6313, 14.6313, 14.6275, 14.6216, 14.6141, 14.6047, 14.6016])

Ps3 = np.array([440.51, 435.21, 421.115, 405.215, 374.09, 349.655, 324.035, 299.7])

P49Q56 = np.array([7.67128, 7.56462, 7.30442, 7.01989, 6.48459, 6.04978, 5.60278, 5.17175])  # corrected

Pt49 = P49Q56 * Pt56

Pt54 = np.array([112.243, 110.68, 106.873, 102.683, 94.815, 88.4119, 81.8269, 75.5156])

Pt17 = np.array([25.4609, 25.6071, 25.4679, 25.1292, 24.3442, 23.6164, 22.7956, 21.9702])

W2 = np.array([1785.86, 1772.57, 1752.63, 1723.51, 1659.13, 1599.55, 1533.94, 1463.94])  # in pps = * 0.45359237
# corrected

T5SD = [1210.95, 1205.96, 1193.27, 1175.34, 1137.46, 1106.25, 1072.37, 1040.37]

""" The following conversions are carried out below
# lbf     to kN      for thrust
# pph     to kg/s    for the fuel flow
# pps     to kg/s    for the mass flow
# Psi     to bar     for the pressure
# Deg C   to Kelvin  for the temperature
"""

Fgdp = np.interp(N1k_rated, np.flip(N1), np.flip(Fg)) * 0.00444822162
N2dp = np.interp(N1k_rated, np.flip(N1), np.flip(N1))
T2dp = np.interp(N1k_rated, np.flip(N1), np.flip(T2)) + 273.15
rhum_dp = np.interp(N1k_rated, np.flip(N1), np.flip(rHum))
Ps0_dp = np.interp(N1k_rated, np.flip(N1), np.flip(Ps0)) * 0.0689475729  # same as barometric pressure (approx)
Pt10dp = np.interp(N1k_rated, np.flip(N1), np.flip(Pt10)) * 0.0689475729
T49dp = np.interp(N1k_rated, np.flip(N1), np.flip(T49)) + 273.15
T3dp = np.interp(N1k_rated, np.flip(N1), np.flip(T3)) + 273.15
T25dp = np.interp(N1k_rated, np.flip(N1), np.flip(T25)) + 273.15  # a bit unsure about this
Wfdp = np.interp(N1k_rated, np.flip(N1), np.flip(Wf)) * 0.0001259979
WFdp, _ = correction(1, Wfdp, Pt10dp, T2dp, True)
Ps25dp = np.interp(N1k_rated, np.flip(N1), np.flip(Ps25)) * 0.0689475729
Pt56dp = np.interp(N1k_rated, np.flip(N1), np.flip(Pt56)) * 0.0689475729  # same as AVPBAR
PS3dp = np.interp(N1k_rated, np.flip(N1), np.flip(Ps3)) * 0.0689475729
Pt49dp = np.interp(N1k_rated, np.flip(N1), np.flip(Pt49)) * 0.0689475729
Pt54dp = np.interp(N1k_rated, np.flip(N1), np.flip(Pt54)) * 0.0689475729
Pt17dp = np.interp(N1k_rated, np.flip(N1), np.flip(Pt17)) * 0.0689475729
W2dp = np.interp(N1k_rated, np.flip(N1), np.flip(W2)) * 0.45359237
W2dp, _ = correction(1, W2dp, Pt10dp, T2dp, True)

true_val_DP_CF6 = np.array([Ps25dp, T25dp, Pt17dp, T3dp, PS3dp, Pt49dp, T49dp, Pt56dp, Fgdp])

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
    R  = 287
    Cp = 1000
    Pdyn = (Pt - Ps) * 1e5
    rho  = (Ps*1e5/R + Pdyn/Cp)/Tt
    V    = np.sqrt(Pdyn*2/rho)
    W    = A*rho*V
    print(V, rho, W)
    return W

#
# import pandas as pd
# import numpy as np
#
# def loadData():
#     # ====================
#     # Load data from excel
#     # ====================
#
#     path = "C:\\Users\\Shivan\\OneDrive - Delft University of Technology\\Thesis"
#
#     filenameRaw = "GEnx baseline data"
#     filenameC = "GEnx baseline data C"
#     filename = "genx_correlation_pc run5"
#
#     # Load correlation report data
#     correlationDataRaw_excel = pd.read_excel(path + "\\" + filenameRaw + ".xlsx", skiprows=1, index_col=0)
#     correlationDataRawC_excel = pd.read_excel(path + "\\" + filenameC + ".xlsx", skiprows=0, index_col=0)
#     correlationData_excel = pd.read_excel(path + "\\" + filename + ".xls ", skiprows=[1, 2],
#                                           index_col=0, sheet_name=1)
#     correlationData_excel = correlationData_excel.loc[:, ~correlationData_excel.columns.str.contains('^Unnamed')]
#
#     return correlationDataRaw_excel, correlationDataRawC_excel, correlationData_excel
#
#
# _, _, datGEnx = loadData()
#
# DPgenxAll = datGEnx['PC_TO_1B7475_2']
#
# T12dp = DPgenxAll.loc['T12_SEL_ChA'] + 273.15
# T2dp = DPgenxAll.loc['TT2'] + 273.15
# TT3dp = DPgenxAll.loc['T3_SEL_ChA'] + 273.15
# TT49dp = DPgenxAll.loc['EGT_SEL_ChA'] + 273.15
# Pt0dp = DPgenxAll.loc['P0_SEL_ChA'] * 0.0689475729
# Ps10dp = DPgenxAll.loc['PS10W_Avg'] * 0.0689475729
# Pt10dp = DPgenxAll.loc['PT10_Avg'] * 0.0689475729
# Pt2dp = DPgenxAll.loc['PT2_SEL_ChA'] * 0.0689475729
# Ps25dp = DPgenxAll.loc['PS25_ChA'] * 0.0689475729
# Ps3dp = DPgenxAll.loc['PS3_SEL_ChA'] * 0.0689475729
# N1dp = DPgenxAll.loc['N1']
# N1KSDdp = DPgenxAll.loc['N1KSD']
# N2dp = DPgenxAll.loc['N2']
# Fndp = DPgenxAll.loc['FNnet'] * 0.004448
# Rhumdp = DPgenxAll.loc['RHUM']
# Wf = DPgenxAll.loc['WF'] * 0.0001259979
# W2dp = 1179.75  # massflow(Ps10dp, Pt10dp, T2dp)  # maybe not possible with total?
# W2dp, _ = correction(1, W2dp, Pt10dp, T2dp, True)
#
# true_val_DP_GEnx = []
#
# massflow(Ps10dp, Pt10dp, T2dp)
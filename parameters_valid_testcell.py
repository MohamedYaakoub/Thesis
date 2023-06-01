"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 8/18/2022 1:01 PM  
"""
import pandas as pd
import numpy as np
import os
from parameters import correction, datGEnxC

path = os.getcwd()
Foldername     = "DataDocs"  # data folder name
filenameTC     = "956609 KLM Engine tested on September 20, 2017"

def loaddataTc():
    # ====================
    # Load data from excel
    # ====================

    # Load correlation report data

    correlationData_excel = pd.read_excel(path + "\\" + Foldername + "\\" + filenameTC + ".xlsx", skiprows=[2, 3],
                                          index_col=0, header=0)
    correlationData_excel = correlationData_excel.loc[:, ~correlationData_excel.columns.str.contains('^Unnamed')]

    return correlationData_excel

dataTc = loaddataTc()

dataTc = dataTc.dropna()

dataTc = dataTc["take off"]

# T12dpgx = dataTc.loc['T12_SEL_ChA'] + 273.15
T2dpgx   = dataTc.loc['TT2'] + 273.15
TT3dpgx  = dataTc.loc['T3_SEL_ChA'] + 273.15
TT25dpgx = dataTc.loc['T25_SEL_ChA'] + 273.15
TT49dpgx = dataTc.loc['EGT_SEL_ChB'] + 273.15
Pt10dpgx = dataTc.loc['P0_SEL_ChA'] * 0.0689475729
Ps10dpgx = dataTc.loc['PS10W_Avg'] * 0.0689475729
Pt2dpgx  = dataTc.loc['PT2_SEL_ChA'] * 0.0689475729
# Ps25dpgx = dataTc.loc['PS25_ChA'] * 0.0689475729
Ps3dpgx  = dataTc.loc['PS3_SEL_ChA'] * 0.0689475729
N1dpgx   = dataTc.loc['N1']
N1KSDdpgx = dataTc.loc['N1KSD']  # to read from table
N2dpgx    = dataTc.loc['N2']
N1pdpgx   = dataTc.loc['N1_SEL_ChA_%']
N2pdpgx   = dataTc.loc['N2_SEL_ChA_%']
Fndpgx    = dataTc.loc['FNnet'] * 0.004448
Rhumdpgx  = dataTc.loc['RHUM']
Wfdpgx    = dataTc.loc['WF'] * 0.0001259979

W2dpgx = np.interp(N1KSDdpgx, np.flip(datGEnxC["ATC.N1"]), np.flip(datGEnxC["ATC.W2A"])) * 0.45359237
W2dpgx, _ = correction(1, W2dpgx, Pt10dpgx, T2dpgx, True)

true_val_DP_GEnx = np.array([TT25dpgx, TT3dpgx, Ps3dpgx, TT49dpgx, Fndpgx])

g     = 1.4
Ts    = T2dpgx/(Pt10dpgx/Ps10dpgx)**((g-1)/g)
Mach  = np.sqrt((T2dpgx/Ts - 1)/(0.5*(g - 1)))
rho   = 10**5*Ps10dpgx/(287*Ts)
v     = Mach * np.sqrt(g*287*Ts)
Mdot  = rho*v*5.5864

class params2:
    """Parameters for the GEnx"""

    def __init__(self):
        self.inputs = {
            # Inlet Module
            'dot_w': Mdot,
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
        self.RmHPTin  = 0.392183952
        self.RmHPTout = 0.368712109
        self.RmLPTin  = 0.547234438
        self.RmLPTout = 0.547234438
        self.RmLPT    = 0.649625787
        # HPC inlet area
        self.HPCinlA = 0.358647582
        self.HPCoutA = 0.082344939
        self.OPR = 46.3  # from https://www.geaviation.com/commercial/engines/genx-engine for the -1B74/75 (787-9)


pGEnx = params2()
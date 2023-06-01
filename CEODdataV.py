"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 8/19/2022 1:16 PM  
"""
import pandas as pd
import numpy as np
import os
# import dtale
from parameters import correction
from matplotlib import pyplot as plt
import pickle

path = os.getcwd()
Foldername = "DataDocs"  # data folder name
filenameCEOD   = "170929-154715-KLM897____-EHAMZBAA-KL_PH-BHA-2-956609-W007FFD.zip"
filenameCEOD1  = "171004-221611-KLM872____-VIDPEHAM-KL_PH-BHA-2-956609-W007FFD.csv"
filenameCEOD2  = "171021-193457-KLM893____-EHAMZSPD-KL_PH-BHA-2-956609-W007FFD.csv"
filenameCEOD3  = "171010-011446-KLM868____-RJBBEHAM-KL_PH-BHA-2-956609-W007FFD.zip"
# filenameCEOD4  = "171028-230032-KLM214____-CYVREHAM-KL_PH-BHA-2-956609-W007FFD.zip"
filenameCEOD5  = "171208-172732-KLM655____-EHAMKMSP-KL_PH-BHA-2-956609-W007FFD.zip"
filenameCEOD6  = "171202-083810-KLM749____-EHAMSKBO-KL_PH-BHA-2-956609-W007FFD.zip"
filenameCEOD7  = "171125-044447-KLM874____-VCBIEHAM-KL_PH-BHA-2-956609-W007FFD.zip"
filenameCEOD8  = "171215-053250-KLM898____-ZBAAEHAM-KL_PH-BHA-2-956609-W007FFD.zip"
filenameCEOD9  = "180102-235649-KLM749____-SKCGEHAM-KL_PH-BHA-2-956609-W007FFD.zip"
filenameCEOD10 = "180120-232801-KLM749____-SKCGEHAM-KL_PH-BHA-2-956609-W007FFD.zip"

name_list      = [filenameCEOD, filenameCEOD1, filenameCEOD2, filenameCEOD3, filenameCEOD5, filenameCEOD6,
                  filenameCEOD7, filenameCEOD8, filenameCEOD9, filenameCEOD10]

ToExtractParams = [
    # "FADEC Bit-packed Trigger Word : CLIMB CONDITIONS DETECTED",
    "Offset",
    "Selected Core Compartment Cooling (CCC) Valve Position (%)",
    "Selected CAI (Cowl Anti Ice) Bleed Config",
    "Selected Booster Anti-Ice (BAI) Pressure (PSIA)",
    "Selected Transient Bleed Valve (TBV) Position (%)",
    "Selected Variable Bleed Valve (VBV) Position (%)",
    "Selected Variable Stator Vane (VSV) Position (%)",
    "Selected HP Turbine Active Clearance Control Valve Position (%)",
    "Selected LP Turbine Active Clearance Control Valve Position (%)",
    "Total Engine Horsepower Extraction (HP)",
    "Selected Mass Fuel Flow (PPH)",
    "Average Gas Temperature at Station 25 (DEG_C)",
    "Calculated Ambient Temperature (DEG_C)",
    "Selected Compressor Delay Total Temperature (DEG_C)",  # TT3
    "Selected Compressor Discharge Static Pressure (PSIA)",
    "Selected Ambient Static Pressure (PSIA)",
    "Selected Exhaust Gas Temperature (DEG_C)",
    "Selected HP Comp Inlet Total Temperature (DEG_C)",
    "Selected PT2 Pressure (PSIA)",
    "Selected Total Temperature at Station 12 (DEG_C)",
    "Selected Fan Speed (%)",
    "Corrected Fan Speed to Station 12 (%)",
    "Selected Core Speed (%)",
    "Corrected Core Speed to Station 25 (%)",
    "Core Speed Rate of Change (%N2/SEC)",
    "Altitude based on P0 (FT)",
    "UVL_FLIGHTPHS",
    "Selected Mach Number (MACH)",
    "Pitch Angle from Aircraft (DEGREES)",
    "Roll Angle from Aircraft (DEGREES)",
    "Calculated Core Airflow (PPS)",
    # "Calculated Fuel to Air Ratio"
    'Header: Start Date'
]

def loadCEOD_GEnx(filenameCEOD):
    # ====================
    # Load data from excel
    # ====================

    # load data
    correlationData_excel = pd.read_csv(path + "\\" + Foldername + "\\" + filenameCEOD, skiprows=1,
                                        index_col=False, header=0, dtype=np.float32, usecols=ToExtractParams)
    return correlationData_excel

data   = pd.DataFrame(columns=ToExtractParams)
dataCL = pd.DataFrame(columns=ToExtractParams)

for file in name_list:
    datai   = loadCEOD_GEnx(file)

    # set all nan flight date entries to actual flight date
    dateFL  = datai['Header: Start Date'][0]
    datai["Header: Start Date"][:] = dateFL

    data    = pd.concat([data, datai])
    dataCLi = datai[(datai["UVL_FLIGHTPHS"] == 6)]
    dataCLi = dataCLi.tail(50)
    dataCL  = pd.concat([dataCL, dataCLi])


# drop rows with nan and reset column index
data   = data.dropna(axis=0).reset_index(drop=True)
dataCL = dataCL.dropna(axis=0).reset_index(drop=True)
# find steady state points

dataCL = dataCL[(dataCL["Core Speed Rate of Change (%N2/SEC)"] > - 0.1)
               & (dataCL["Core Speed Rate of Change (%N2/SEC)"] < 0.1)
               & (dataCL["Selected HP Turbine Active Clearance Control Valve Position (%)"] < 20)
               & (dataCL["Selected LP Turbine Active Clearance Control Valve Position (%)"] < 55)
               & (dataCL["Selected CAI (Cowl Anti Ice) Bleed Config"] > 0.1)
               & (dataCL["Altitude based on P0 (FT)"] < 35000)
               & (dataCL["Altitude based on P0 (FT)"] > 28000)
               & (np.round(dataCL["Corrected Fan Speed to Station 12 (%)"], 1) != 101.8)
               ]

# dataCL = dataCL.tail(30*len(name_list))

dataTO = data[(data["Core Speed Rate of Change (%N2/SEC)"] > - 0.2)
              & (data["Core Speed Rate of Change (%N2/SEC)"] < 0.2)
              & (data["UVL_FLIGHTPHS"] == 4)
              # & (data["Selected Transient Bleed Valve (TBV) Position (%)"] < 1)
              # & (data["Selected Variable Bleed Valve (VBV) Position (%)"] < -0.1)
              # & (data["Selected Variable Stator Vane (VSV) Position (%)"] < 90)
              & (data["Selected HP Turbine Active Clearance Control Valve Position (%)"] < 25)
              & (data["Selected LP Turbine Active Clearance Control Valve Position (%)"] < 48)
              & (data["Selected CAI (Cowl Anti Ice) Bleed Config"] < 0.1)
              & (data["Pitch Angle from Aircraft (DEGREES)"] > 0.5)
              ]

dataCR = data[(data["Core Speed Rate of Change (%N2/SEC)"] > - 0.1)
              & (data["Core Speed Rate of Change (%N2/SEC)"] < 0.1)
              & (data["UVL_FLIGHTPHS"] == 7)
              & (data["Selected Transient Bleed Valve (TBV) Position (%)"] < 1)
              & (data["Selected Booster Anti-Ice (BAI) Pressure (PSIA)"] < 10)
              & (data["Selected CAI (Cowl Anti Ice) Bleed Config"] > 0.1)
              & (data["Selected Booster Anti-Ice (BAI) Pressure (PSIA)"] < 5)
              & (data["Selected Core Compartment Cooling (CCC) Valve Position (%)"] > 99)
              & (data["Selected Variable Bleed Valve (VBV) Position (%)"] < -0.08)
              & (data["Selected Variable Bleed Valve (VBV) Position (%)"] > -0.19)
              & (data["Selected Variable Stator Vane (VSV) Position (%)"] < 80)
              & (data["Selected HP Turbine Active Clearance Control Valve Position (%)"] < 70)
              & (data["Selected HP Turbine Active Clearance Control Valve Position (%)"] > 40)
              & (data["Selected LP Turbine Active Clearance Control Valve Position (%)"] > 70)
              ]

ind_drop = np.array([], dtype=np.int32)

unique_n1C = np.unique(dataCR['Corrected Fan Speed to Station 12 (%)'].values.round(decimals=0))

unique_Alt = np.unique(dataCR["Altitude based on P0 (FT)"].values.round(decimals=-3))

for i in range(len(unique_Alt)):
    # ind_N1c = np.array([], dtype=np.int32)
    indicesA = np.where(dataCR["Altitude based on P0 (FT)"].values.round(decimals=-3) == unique_Alt[i])[0]
    for j in range(len(unique_n1C)):
        indicesN = \
        np.where(dataCR['Corrected Fan Speed to Station 12 (%)'].values.round(decimals=0) == unique_n1C[j])[0]
        indices = np.intersect1d(indicesA, indicesN)
        if len(indices) >= 5:
            ind_drop = np.append(ind_drop, indices[:5])
        else:
            ind_drop = np.append(ind_drop, indices[:len(indices)+1])

dataCR2 = dataCR.loc[dataCR.index[ind_drop]]

GEnx_ODL, GEnx_OD_trueL, N1cCEODL, N1CEODL = [], [], [], []

for i, dataI in enumerate([dataTO, dataCL, dataCR2]):  #

    data = dataI

    g = 1.4
    isentr = 1 + (g - 1) * 0.5 * data["Selected Mach Number (MACH)"] ** 2
    TsComp = (data["Selected Total Temperature at Station 12 (DEG_C)"] + 273.15) / isentr
    PsComp = (data["Selected PT2 Pressure (PSIA)"] * 0.0689475729) / isentr ** (g / (g - 1))

    # N1c = data["Selected Fan Speed (%)"]/np.sqrt((data["Selected Total Temperature at Station 12 (DEG_C)"] + 273.15)/288)
    N1cdp = 94.96 / (np.sqrt((287.5*275.55) / (288.15 * 287.05)))  # old model (un-calib)
    N1cdp2 = 96.96/(np.sqrt((288.18*285.6)/(288.15*287.05)))  # new model (calib) + martijn model
    # N1c   = 100*data["Selected Fan Speed (%)"]\
    #         / (np.sqrt((288*(data["Selected Total Temperature at Station 12 (DEG_C)"] + 273.15))/(288.15*287.05))) / N1cdp
    N1c   = 100*data["Selected Fan Speed (%)"]\
            / (np.sqrt((288*(data["Selected Total Temperature at Station 12 (DEG_C)"] + 273.15))/(288.15*287.05))) / N1cdp2

    GEnx_OD_true = np.vstack((data["Selected Fan Speed (%)"],
                              N1c,
                              data["Corrected Fan Speed to Station 12 (%)"],
                              data["Selected HP Comp Inlet Total Temperature (DEG_C)"] + 273.15,
                              data["Selected Compressor Delay Total Temperature (DEG_C)"] + 273.15,
                              data["Selected Compressor Discharge Static Pressure (PSIA)"] * 0.0689475729,
                              data["Selected Exhaust Gas Temperature (DEG_C)"] + 273.15,
                              data["Selected Mass Fuel Flow (PPH)"] * 0.0001259979,
                              data["Selected Core Speed (%)"]
                              # data["Calculated Core Airflow (PPS)"] * 0.45359237
                              # data["Selected PT2 Pressure (PSIA)"] * 0.0689475729,
                              # data["Selected Total Temperature at Station 12 (DEG_C)"] + 273.15
                              )).T

    GEnx_OD_true = GEnx_OD_true[GEnx_OD_true[:, 1].argsort()]

    GEnx_OD_true = np.flip(GEnx_OD_true, axis=0)
    N1CEOD       = GEnx_OD_true[:, 0]
    N1cCEOD      = GEnx_OD_true[:, 2]
    GEnx_OD_true = GEnx_OD_true[:, 3:]


    GEnx_OD = np.vstack((N1c,
                         # data["Corrected Fan Speed to Station 12 (%)"],
                         PsComp,
                         TsComp,
                         # data["Selected Ambient Static Pressure (PSIA)"] * 0.0689475729,
                         # data["Calculated Ambient Temperature (DEG_C)"] + 273.15,
                         data["Selected Mach Number (MACH)"],
                         data["Total Engine Horsepower Extraction (HP)"] * 0.745699872)).T

    GEnx_OD = GEnx_OD[GEnx_OD[:, 0].argsort()]
    GEnx_OD = np.flip(GEnx_OD, axis=0)
    Cv      = np.full((GEnx_OD.shape[0], 1), 0.98) if i==0 else np.full((GEnx_OD.shape[0], 1), 1)
    # GEnx_OD = np.hstack((GEnx_OD, Cv))

    GEnx_ODL.append(GEnx_OD)
    GEnx_OD_trueL.append(GEnx_OD_true)
    N1cCEODL.append(N1cCEOD)
    N1CEODL.append(N1CEOD)

pickle.dump([GEnx_ODL, GEnx_OD_trueL, N1cCEODL], open("CEOD_input.p", "wb"))

# %%
data    = data.sort_values("Corrected Fan Speed to Station 12 (%)", ascending=True)

dataplt = data[["Corrected Fan Speed to Station 12 (%)",
                "Selected Core Compartment Cooling (CCC) Valve Position (%)",
                "Selected CAI (Cowl Anti Ice) Bleed Config",
                "Selected Booster Anti-Ice (BAI) Pressure (PSIA)",
                "Selected Transient Bleed Valve (TBV) Position (%)",
                "Selected Variable Bleed Valve (VBV) Position (%)",
                "Selected Variable Stator Vane (VSV) Position (%)",
                "Selected HP Turbine Active Clearance Control Valve Position (%)",
                "Selected LP Turbine Active Clearance Control Valve Position (%)",
                "Altitude based on P0 (FT)",
                "Selected Mach Number (MACH)",
                "Core Speed Rate of Change (%N2/SEC)",
                "Calculated Ambient Temperature (DEG_C)"]]

dataplt2 = data[["Corrected Fan Speed to Station 12 (%)",
                "Selected Fan Speed (%)",
                "Selected Ambient Static Pressure (PSIA)",
                "Selected Core Speed (%)",
                "Corrected Core Speed to Station 25 (%)",
                "Roll Angle from Aircraft (DEGREES)",
                "Pitch Angle from Aircraft (DEGREES)",
                "Selected Mass Fuel Flow (PPH)",
                "Average Gas Temperature at Station 25 (DEG_C)",
                "Selected Compressor Delay Total Temperature (DEG_C)",  # TT3
                "Selected Compressor Discharge Static Pressure (PSIA)",
                "Selected Exhaust Gas Temperature (DEG_C)",
                "Selected HP Comp Inlet Total Temperature (DEG_C)",
                "Calculated Core Airflow (PPS)",
                "Selected PT2 Pressure (PSIA)",
                "Selected Total Temperature at Station 12 (DEG_C)"
                ]]

if __name__ == '__main__':

    #%%
    colnames = list(dataplt.columns)
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(25, 15))

    # unpack all the axes subplots
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        dataplt.plot(x="Corrected Fan Speed to Station 12 (%)", y=colnames[i+1], kind='scatter', ax=ax, label=colnames[i+1])
        ax.legend(loc='best')
        ax.set_ylabel("")
    plt.tight_layout()
    plt.show()

    colnames = list(dataplt2.columns)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(25, 13))

    # unpack all the axes subplots
    axes = axes.ravel()

    for i, ax in enumerate(axes[:-1]):
        dataplt2.plot(x="Corrected Fan Speed to Station 12 (%)", y=colnames[i+1], kind='scatter', ax=ax, label=colnames[i+1])
        ax.legend(loc='best')
        ax.set_ylabel("")
    plt.tight_layout()
    plt.show()

    flights = len(np.unique(data["Header: Start Date"]))
    print("Flights:", flights)



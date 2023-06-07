"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 8/31/2022 7:43 PM  
"""

import pandas as pd
import numpy as np
import os
from ambiance import Atmosphere
from matplotlib import pyplot as plt
import sys
from GSP_helper import cleanup, runGsp, loadModel


path = os.getcwd()
Foldername   = "DataDocs" + "\\" + "704399 On Wing Data"  # data folder name
filename     = "1346536_B2B Engine v2.csv"


ToExtractParams = [
    "ENGINE_TYPE",       # "CF6-80C2B5F"
    "FLIGHT_PHASE",
    "FLIGHT_DATETIME",   # B2B September 1, 2021
    "Indicated Fan Speed",
    "Corrected Fan Speed",
    "Core Speed",
    "Static Air Temperature",
    "Total Air Temperature",
    "HPC Inlet Total Temp",  # is zero where ZT3 is non zero and for TO + B5F
    "LPT Discharge Total Temp",
    "HPC Inlet Total Pres",
    "HPC Discharge Static Pressure PS3",  # is zero where ZT3 is non zero and for TO + B5F
    "LPT Inlet Pressure",  # is zero where ZT3 is non zero and for TO + B5F
    "Fuel Flow",
    "Fan OGV Discharge Pres",  # is zero where ZT3 is non zero and for TO + B5F
    "EGT",
    "Mach",
    # "ETOPS EGT Margin - Cruise",
    "ZT3",
    "Altitude",
    "FLIGHT_PHASE",
    # "Departure Station",
    # "VBV Position",
    # "VSV Position",
]

def loadSnapshot(filename):
    # ====================
    # Load data from excel
    # ====================

    # load data
    correlationData_excel = pd.read_csv(path + "\\" + Foldername + "\\" + filename, skiprows=0,
                                        index_col=False, header=0, usecols=ToExtractParams)
    return correlationData_excel


data = loadSnapshot(filename)

# data = data[
#             #(data['ZT3'].notna()) &
#             (data["FLIGHT_PHASE"] == "TAKEOFF") &
#             (data["ENGINE_TYPE"] == "CF6-80C2B5F") &
#             (data["Altitude"] < 2000) &
#             # (list(map(lambda x: x.startswith('2021'), data["FLIGHT_DATETIME"]))) &
#             (data["Mach"] > 0.25)]

data   = data.dropna(axis=0).reset_index(drop=True)

N1cdp  = 107.6 / (np.sqrt((288.82 * 292.23) / (288.15 * 287.05)))

N1c = 100 * data["Indicated Fan Speed"] \
      / (np.sqrt(
     (288 * (data["Total Air Temperature"] + 273.15)) / (288.15 * 287.05))) / N1cdp


# CF6_OD_true     = np.vstack((data["HPC Inlet Total Pres"] * 0.0689475729, data["ZT3"] + 273.15,
#                              data["EGT"] + 273.15, data["LPT Discharge Total Temp"] + 273.15,
#                              data['Fuel Flow'] * 0.0001259979, data["Core Speed"])).T

CF6_OD_true     = np.vstack((data["HPC Inlet Total Pres"] * 0.0689475729, data["HPC Inlet Total Temp"] + 273.15,
data["Fan OGV Discharge Pres"] * 0.0689475729, data["HPC Discharge Static Pressure PS3"] * 0.0689475729,
                             data["LPT Inlet Pressure"] * 0.0689475729,
                             data["EGT"] + 273.15, data["LPT Discharge Total Temp"] + 273.15,
                             data['Fuel Flow'] * 0.0001259979, data["Core Speed"])).T

Sealevel = Atmosphere(0)
PSL      = Sealevel.pressure/100000

Alts     = Atmosphere(list(data["Altitude"]))

Pamb     = Alts.pressure/100000


# CF6_OD_true     = CF6_OD_true[:, 1:]
CF6_OD          = np.vstack((N1c, Pamb, data["Static Air Temperature"] + 273.15, data["Mach"])).T

modelName  = "OffDesignValid 2.mxl"

gspdll = loadModel(0, modelName)

inputs_list = ["N1", "P0", "T0", "Mach"]
# output_list = ["P25", "TT3", "TT49", "TT5", "Wf", "N2"]

output_list = ["P25", "TT25", "Ps14", "Ps3", "P49", "TT49", "TT5", "Wf", "N2"]

# %%
def compute_error(inputDat, trueVal):
    y_sim        = np.array(runGsp(gspdll, inputDat, output_list))
    change   = (trueVal - y_sim)/(trueVal+0.000001)
    meanE     = 100*np.sqrt(np.mean(change**2, axis=0))
    return meanE, change

mean, change  = compute_error(CF6_OD, CF6_OD_true)

cmap = plt.get_cmap('tab20')
clist = cmap(np.linspace(0, 1, len(output_list)))
plt.figure()

#%%

for i in range(len(output_list)):
    plt.scatter(CF6_OD[:, 0], 100 * change[:, i], label=output_list[i])
plt.xlabel('Corrected Fan Speed [%]')
plt.ylabel('Error (CEOD - GSP) [%]')
plt.legend(loc='lower center')
plt.show()

#%%
cleanup(gspdll)

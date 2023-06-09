import numpy as np
import pandas as pd
import os
import pickle
from matplotlib import pyplot as plt
from helper_variables import ToExtractParams

print(f'There are currently {len(os.listdir("CEOD_GEnx/csv files"))} csv files in the CEOD_GEnx/csv files directory')
N2_boxplot = []
P0_boxplot = []
T0_boxplot = []
Ma_boxplot = []
HP_boxplot = []
alt_list = []
time_list = []

counter = 0
for file in os.listdir("CEOD_GEnx/csv files"):
    counter += 1
    print(counter)
    # file = '200408-203904-KLM168____-KATLEHAM-KL_PH-BHA-2-956609-W010FFD.csv'
    # file = '160801-040522-KLM884____-ZSAMEHAM-KL_PH-BHA-2-956609-W007FFD.csv'
    data = pd.read_csv("CEOD_GEnx/csv files/" + file, skiprows=1, index_col=False, header=0, dtype=np.float32,
                       usecols=ToExtractParams)
    data["Header: Start Date"][:] = data['Header: Start Date'][0]
    dataCL = data[(data["UVL_FLIGHTPHS"] == 5) | (data["UVL_FLIGHTPHS"] == 6)]
    #     dataCL = dataCL.tail(50)
    #     print(dataCL.shape)

    data = data.dropna(axis=0).reset_index(drop=True)
    dataCL = dataCL.dropna(axis=0).reset_index(drop=True)
    #     dataCL = dataCL.tail(50)
    #     dataCL = dataCL.dropna(axis=0).reset_index(drop=True)

    dataCL = dataCL[(dataCL["Core Speed Rate of Change (%N2/SEC)"] > - 0.1)
                    & (dataCL["Core Speed Rate of Change (%N2/SEC)"] < 0.1)
                    & np.insert(np.abs(np.diff(dataCL["Altitude based on P0 (FT)"])) > 15, 0, True)
                    # & (dataCL["Selected HP Turbine Active Clearance Control Valve Position (%)"] < 20)
                    # & (dataCL["Selected LP Turbine Active Clearance Control Valve Position (%)"] < 55)
                    # & (dataCL["Selected CAI (Cowl Anti Ice) Bleed Config"] > 0.1)
                    # & (dataCL["Altitude based on P0 (FT)"] < 35000)
                    # & (dataCL["Altitude based on P0 (FT)"] > 28000)
                    # & (np.round(dataCL["Corrected Fan Speed to Station 12 (%)"], 1) != 101.8)
                    ]

    dataTO = data[(data["Core Speed Rate of Change (%N2/SEC)"] > - 0.1)
                  & (data["Core Speed Rate of Change (%N2/SEC)"] < 0.1)
                  & (data["UVL_FLIGHTPHS"] == 4)
                  # & (data["Selected Transient Bleed Valve (TBV) Position (%)"] < 1)
                  # & (data["Selected Variable Bleed Valve (VBV) Position (%)"] < -0.1)
                  # & (data["Selected Variable Stator Vane (VSV) Position (%)"] < 90)
                  # #               & (data["Selected HP Turbine Active Clearance Control Valve Position (%)"] < 25)
                  # & (data["Selected LP Turbine Active Clearance Control Valve Position (%)"] < 48)
                  # & (data["Selected CAI (Cowl Anti Ice) Bleed Config"] < 0.1)
                  # & (data["Pitch Angle from Aircraft (DEGREES)"] > 0.5)
                  ]

    dataCR = data[(data["Core Speed Rate of Change (%N2/SEC)"] > - 0.1)
                  & (data["Core Speed Rate of Change (%N2/SEC)"] < 0.1)
                  & (data["UVL_FLIGHTPHS"] == 7)
                  # & (data["Selected Transient Bleed Valve (TBV) Position (%)"] < 1)
                  # & (data["Selected Booster Anti-Ice (BAI) Pressure (PSIA)"] < 10)
                  # & (data["Selected CAI (Cowl Anti Ice) Bleed Config"] > 0.1)
                  # & (data["Selected Booster Anti-Ice (BAI) Pressure (PSIA)"] < 5)
                  # & (data["Selected Core Compartment Cooling (CCC) Valve Position (%)"] > 99)
                  # & (data["Selected Variable Bleed Valve (VBV) Position (%)"] < -0.08)
                  # & (data["Selected Variable Bleed Valve (VBV) Position (%)"] > -0.19)
                  # & (data["Selected Variable Stator Vane (VSV) Position (%)"] < 80)
                  # & (data["Selected HP Turbine Active Clearance Control Valve Position (%)"] < 70)
                  # & (data["Selected HP Turbine Active Clearance Control Valve Position (%)"] > 40)
                  # & (data["Selected LP Turbine Active Clearance Control Valve Position (%)"] > 70)
                  ]
    #     print(dataCL)
    # print("TAke off ", dataTO.shape)
    # print("Climb ", dataCL.shape)
    # print("Cruise ", dataCR.shape)

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
                ind_drop = np.append(ind_drop, indices[:len(indices) + 1])

    dataCR2 = dataCR.loc[dataCR.index[ind_drop]]

    GEnx_ODL, GEnx_OD_trueL, N1cCEODL, N1CEODL = [], [], [], []

    for i, dataI in enumerate([dataTO, dataCL, dataCR2]):  #

        data = dataI

        g = 1.4
        isentr = 1 + (g - 1) * 0.5 * data["Selected Mach Number (MACH)"] ** 2
        TsComp = (data["Selected Total Temperature at Station 12 (DEG_C)"] + 273.15) / isentr
        PsComp = (data["Selected PT2 Pressure (PSIA)"] * 0.0689475729) / isentr ** (g / (g - 1))

        # N1c = data["Selected Fan Speed (%)"]/np.sqrt((data["Selected Total Temperature at Station 12 (DEG_C)"] + 273.15)/288)
        N1cdp = 94.96 / (np.sqrt((287.5 * 275.55) / (288.15 * 287.05)))  # old model (un-calib)
        N1cdp2 = 96.96 / (np.sqrt((288.18 * 285.6) / (288.15 * 287.05)))  # new model (calib) + martijn model
        # N1c   = 100*data["Selected Fan Speed (%)"]\
        #         / (np.sqrt((288*(data["Selected Total Temperature at Station 12 (DEG_C)"] + 273.15))/(288.15*287.05))) / N1cdp
        N1c = 100 * data["Selected Fan Speed (%)"] \
              / (np.sqrt(
            (288 * (data["Selected Total Temperature at Station 12 (DEG_C)"] + 273.15)) / (288.15 * 287.05))) / N1cdp2

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
        N1CEOD = GEnx_OD_true[:, 0]
        N1cCEOD = GEnx_OD_true[:, 2]
        GEnx_OD_true = GEnx_OD_true[:, 3:]

        GEnx_OD = np.vstack((N1c,
                             # data["Corrected Fan Speed to Station 12 (%)"],
                             PsComp,
                             TsComp,
                             # data["Selected Ambient Static Pressure (PSIA)"] * 0.0689475729,
                             # data["Calculated Ambient Temperature (DEG_C)"] + 273.15,
                             data["Selected Mach Number (MACH)"],
                             data["Total Engine Horsepower Extraction (HP)"] * 0.745699872)).T
                            # data["Altitude based on P0 (FT)"],
                            #  data["Offset"])).T

        GEnx_OD = GEnx_OD[GEnx_OD[:, 0].argsort()]
        GEnx_OD = np.flip(GEnx_OD, axis=0)
        Cv = np.full((GEnx_OD.shape[0], 1), 0.98) if i == 0 else np.full((GEnx_OD.shape[0], 1), 1)
        # GEnx_OD = np.hstack((GEnx_OD, Cv))
        GEnx_ODL.append(GEnx_OD)
        GEnx_OD_trueL.append(GEnx_OD_true)
        N1cCEODL.append(N1cCEOD)
        N1CEODL.append(N1CEOD)

    pickle.dump([GEnx_ODL, GEnx_OD_trueL, N1cCEODL], open("CEOD_GEnx/CEOD_" + file.strip(".csv") + ".p", "wb"))
    # flight_phase = 1
    # N2_boxplot.append(GEnx_ODL[flight_phase][:, 0])
    # P0_boxplot.append(GEnx_ODL[flight_phase][:, 1])
    # T0_boxplot.append(GEnx_ODL[flight_phase][:, 2])
    # Ma_boxplot.append(GEnx_ODL[flight_phase][:, 3])
    # HP_boxplot.append(GEnx_ODL[flight_phase][:, 4])
    # alt_list.append(GEnx_ODL[flight_phase][:, 5])
    # time_list.append(GEnx_ODL[flight_phase][:, 6])
    # print(GEnx_ODL[flight_phase][:, 0].shape)


def filter_outliers(data_array):
    filtered_data = []
    data_before_filtering = 0

    for data in data_array:
        running = True
        print(f'data size at the beginning {len(data)}')
        if len(data) == 0:
            filtered_data.append([])
        else:
            while running:
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                condition = (data >= lower_bound) & (data <= upper_bound)
                if False in condition:
                    data = data[condition]
                else:
                    running = False
            filtered_data.append(data)

    return filtered_data


# plt.boxplot(filter_outliers(N2_boxplot))
# plt.title("N2")
# plt.show()
#
# plt.boxplot(filter_outliers(T0_boxplot))
# plt.title("T0")
# plt.show()
#
# plt.boxplot(filter_outliers(P0_boxplot))
# plt.title("P0")
# plt.show()
#
# plt.boxplot(filter_outliers(Ma_boxplot))
# plt.title("Mach")
# plt.show()
#
# plt.boxplot(filter_outliers(HP_boxplot))
# plt.title("HP")
# plt.show()


# for i in range(len(T0_boxplot)):
#     plt.scatter(N2_boxplot[i], T0_boxplot[i], label=str(i))
#
# plt.legend()
# plt.show()
#
#
# for i in range(len(T0_boxplot)):
#     plt.scatter(time_list[i], alt_list[i])
# plt.ylabel("alt")
# plt.xlabel("offset")
# plt.legend()
# plt.show()

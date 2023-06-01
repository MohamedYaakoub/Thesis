"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 8/26/2022 11:05 AM  
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from GSP_helper import cleanup, runGsp

# sns.get_dataset_names()
# dataset  = sns.load_dataset('diamonds', cache=True, data_home=None,)
# datasetC = dataset.corr()

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesignGEnx Valid Sens.mxl"

# note: these should be identical to the parameters defined in the API module of GSP

inputs_list = ["N1", "P0", "T0", "Mach", "CVc", "CVbp"]
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2", "Wc", "Wbp", "Vc", "Vbp"]

# dump the following to transfer them to the objective function file
pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))
# import the objective function
from OD_function import gspdll

param_val = [97, 0.98, 299, 0.25, 1, 1]

# param_val = [97, 0.22, 215, 0.85, 1, 1]

def sensitivity(selected_k, inputs_list, output_list):
    """
    Perform a sensitivity analyses using a list of selected keys that can be chosen from (and should be identical to)
    the parameters input class.

    :param p:
    :param selected_k:
    :return:
    """
    factor = 0.98
    keys = inputs_list

    indices = [keys.index(selected_k[i]) for i in range(len(selected_k))]

    refrence = np.array(runGsp(gspdll, [param_val], output_list))

    output = []
    for i in indices:
        in_var = param_val.copy()
        in_var[i] = in_var[i] * factor
        results = np.array(runGsp(gspdll, [in_var], output_list))
        diff = [100 * (results[i] - refrence[i]) / (refrence[i]+0.00001) for i in range(len(refrence))]
        output.append(diff[0])

    return output

keys   = ["CVc", "CVbp"]
output = sensitivity(keys, inputs_list, output_list)

cleanup(gspdll)

#%%
listout  = np.array(output).T
df       = pd.DataFrame(listout, columns=["CVc", "CVbp"], index=output_list)
plt.figure(figsize=(4, 5))
sns.heatmap(df,  annot=True, cbar=True, cmap='coolwarm')
plt.show()
#%%
def barC(outputval, selected_k, params_out, y_name):
    plt.rcParams['figure.dpi'] = 500
    outp_length = len(outputval[0]) if isinstance(outputval[0], list) else len(outputval)
    len_k       = len(selected_k) if len(selected_k) == 3 else 1
    w = 6  # in inch the width of the figure was 6
    h = 3  # height in inch was 3
    r = np.arange(outp_length)
    r = r * w / outp_length
    width = 0.18  # the bar with was0.18
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    colorl  = ['coral', 'goldenrod', 'royalblue']

    for i in range(len_k):
        label      = selected_k[i] if len(selected_k) == 3 else selected_k
        rec = ax.bar(r + width*i, np.round(outputval[i], 1), color=colorl[i], width=width, edgecolor=colorl[i], label=label,
               tick_label=outputval[i])
        # ax.bar_label(rec, padding=3)
    ax.yaxis.grid()  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest
    plt.xlabel("Parameters")
    plt.ylabel(y_name)
    # plt.title("Sensitivity ")
    plt.ylim(0, 8.5)
    plt.xticks(r + width * len_k / 2.6, params_out)  # was 2.6
    plt.yticks(np.arange(9))
    plt.legend(loc='upper right')  # lower
    fig.tight_layout()
    plt.margins(y=0.1)
    plt.show()

unC= np.array([9.099999609651045, 2.1549919990126183, 1.5752182041381115, 21.391284427002535, 0.9101106864202795, 0.8802054162523129, 0.8971994049002746, 0.8969494640950004, 0.8987228185093691, 0.9306593868837868, 0.9450679400258979])
C  = np.array([[9.099579445479264, 2.1915909089517114, 1.5610148690902443, 20.95390258937267, 0.9155096296918898, 0.8874570146206988, 0.8944829465164816, 0.9075182101371015, 0.9074727286671787, 0.9376350508985418, 0.929797993929104]])

# print(100*(C-unC)/C)

Mto = 0.8
Mcr = 1
g   = 1.4
Abp  =  3.0968

def Wred(M,A,g):

    return A * M * np.sqrt(g) / (1 + (g - 1) * 0.5 * M**2)**((g+1)/(2 * (g-1)))

Wto = Wred(Mto, Abp, g)
Wcr = Wred(Mcr, Abp, g)
Wchange = 100*(Wcr-Wto)/Wto





# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 2021

@author: Shivan Ramdin
"""

import numpy as np
import ctypes as ctypes
from _ctypes import FreeLibrary
import timeit
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution
from parameters_old import params, params2

# the parameters for the sensitivity analyses (select from input parameters)

selected_k1  = ['fan_efC', 'fan_efBp', 'fanCore_pr', 'fanBp_pr', 'HPC_pr', 'HPC_ef']
selected_k2  = ['HPT_ef', 'LPT_ef', 'CD_bp', 'CX_bp', 'CD_c', 'CX_c']
# selected_k3  = ['fan_A_Bp', 'fan_A_C', 'HPC_A', 'HPT_A', 'LPT_A']
selected_k4  = ['BPR', 'dot_w', 'CV_c', 'CV_bp']

Engine = 1

# %%
def objfun(inputs, outputs):
    # settings model, these input parameters should be specified in the GSP API
    for i in range(0, len(inputs)):
        gspdll.SetInputControlParameterByIndex(i + 1, ctypes.c_double(inputs[i]))

    gspdll.RunModel(0, 0, 0, 0)

    # this is the output from the model, also specified in GSP API
    output_set = []
    for j in range(1, len(outputs)+1):  # TODO change the variables
        dv = ctypes.c_double(0.)
        gspdll.GetOutputDataParameterValueByIndex(j, ctypes.byref(dv), 0)
        output_set.append(dv.value)
    return output_set


def sensitivity(p, selected_k):
    """
    Perform a sensitivity analyses using a list of selected keys that can be chosen from (and should be identical to)
    the parameters input class.

    :param p:
    :param selected_k:
    :return:
    """
    factor = 1.01
    inputs_dict = p.inputs
    keys = list(inputs_dict.keys())
    values = list(inputs_dict.values())

    indices = [keys.index(selected_k[i]) for i in range(len(selected_k))]

    refrence = objfun(values, p.output_list)

    output = []
    for i in indices:
        in_var = values.copy()
        in_var[i] = in_var[i] * factor
        results_dict = objfun(in_var, p.output_list)
        diff = [100 * (results_dict[i] - refrence[i]) / (refrence[i]+0.00001) for i in range(len(refrence))]
        output.append(diff)
    return output


def plot(outputval, selected_k, params_out):
    outp_length = len(outputval[0])
    w = 12  # in inch the width of the figure
    h = 5   # height in inch
    r = np.arange(outp_length)
    r = r * w / outp_length
    width = 0.2  # the bar with
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    colorl  = ['coral', 'paleturquoise', 'mediumseagreen', 'whitesmoke', 'cornflowerblue', 'darkgrey', 'm', 'y', 'p']

    for i in range(len(selected_k)):
        ax.bar(r + width*i, outputval[i], color=colorl[i], width=width, edgecolor='black', label=selected_k[i],
               tick_label=outputval[i])
    ax.yaxis.grid()  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest
    plt.xlabel("Parameters")
    plt.ylabel("% Change")
    # plt.title("Sensitivity ")
    plt.xticks(r + width * len(selected_k) / 2.2, params_out)
    plt.legend()
    fig.tight_layout()
    plt.show()


# %% loading the model
start = timeit.default_timer()

# load model again for actual run
gspdll = ctypes.cdll.LoadLibrary("GSP.dll")
gspdll.LoadModel("CF6-80C2/DP.mxl", 0) if Engine==0 else gspdll.LoadModel("XML/GEnx-1B_V41.mxl", 0)
gspdll.ConfigureModel()  # no more GSP error messages which disrupt run

# %% runs
if __name__ == "__main__":
    for selected_k in [selected_k1, selected_k2, selected_k4]:

        p      = params() if Engine == 0 else params2()
        output = sensitivity(p, selected_k)

        # %% plot
        plot(output, selected_k, p.output_list)

        end = timeit.default_timer()

        print('Time            : ', end - start)  # optimisation time


    # %%
    gspdll.CloseModel(True)  # Close model, don't show save dialog
    gspdll.FreeAll()  # Free dll before unloading
    FreeLibrary(gspdll._handle)  # Unload dll

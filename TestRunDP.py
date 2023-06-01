"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 4/13/2022 10:33 AM  
"""
import numpy as np
from _ctypes import FreeLibrary
import ctypes
from GSP_helper import loadModel, runGsp, cleanup
from parameters import true_val_DP_GEnx, true_val_DP_CF6, params, params2, PRFanc
import matplotlib.pyplot as plt
from Fletcher_data import loading, etaP_f, etaP_C, flow_coef
from Optimum_param import FPR
from smithChart import rbf

#%%
Engine = 0

# all the parameters
p = params() if Engine == 0 else params2()

# input_data  = [5.00003069,  1.70585793, 13.42233295,  0.90997729,  0.898917,    0.8956728,
#   0.86652054,  0.91741156,  0.94110061,  0.92398091]  # no PR constr (pop = 30)
#
# input_data  = [5.13240091,  1.73052491, 12.92063274,  0.89669106,  0.90727755,  0.90605928,
#   0.88419642, 0.91724176,  0.94912801,  0.91223305]   # best (pop = 25, smoothing is done), Area not modified
#
# input_data  = [5.09517175,  1.72861375, 12.92064442,  0.89594634,  0.90426591,  0.90973621,
#   0.88045182,  0.9177066,   0.94359028,  0.91537075]  # cf6, w smoothing and new area
#
# input_data  = [9.09999741,  2.2186639,   1.57532271, 20.77733072,  0.90145547,  0.91354838,
#   0.92258015,  0.87200075,  0.90792705,  0.96127937,  0.93966459]  # For the GEnx
#
# """ [ 5.09517175  1.72861375 12.92064442  0.89594634  0.90426591  0.90973621
#   0.88045182  0.9177066   0.94359028  0.91537075]"""  # best voor cf6 pop =25, iter 13 + smoothing
#
# """ [ 9.09999741  2.2186639   1.57532271 20.77733072  0.90145547  0.91354838
#   0.92258015  0.87200075  0.90792705  0.96127937  0.93966459] """  # best for the GEnx pop = 25, iter 13 + smoothing



# new results with modified A3


if Engine == 0:
    inputs_list = ["BPR", "FPRbp", "HPCpr", 'HPC_ef', 'fan_efC', 'HPT_ef', 'LPT_ef', 'fan_efD', 'Cx_core', 'Cx_bp']
    old_modelIn = np.array([5.1265, 2.5752, 1.7265, 13.0115, 0.9118, 0.9166, 0.8865, 0.900, 0.9120, 0.9, 0.915])  # fan core PR is added
    output_list = ["TT25", "Ps14", "TT3", "Ps3", "Pt49", "TT49", "T5", "FN", "dH_HPC", "dH_FanC", "dH_FanBp",
                   "A_c", "A_bp", "Eta_fan"]
    input_data  = [5.084086470704756, 1.7280973721923745, 12.97756833279673, 0.8973505330918321, 0.903470456888131,
             0.9082592265792493, 0.8719695048669192, 0.9181738420687525, 0.9188437011863649, 0.9270913176702706]


else:
    inputs_list = ["BPR", "FPRc", "FPRbp", "HPCpr", 'HPC_ef', 'fan_efC', 'HPT_ef', 'LPT_ef', 'fan_efD', 'Cx_core',
                   'Cx_bp']
    old_modelIn = np.array([9.0181, 2.2296, 1.5482, 20.7521, 0.9118, 0.9106, 0.93, 0.94, 0.9248, 0.9, 0.915])  # fan core PR is added
    output_list = ["TT25", "TT3", "Ps3", "TT49", "FN", "dH_HPC", "dH_FanC", "dH_FanBp",
                   "A_c", "A_bp", "Eta_fan", "VxHPTin", "VxHPTout", "VxLPTout", "dH_HPT", "dH_LPT"]
    # Optimised model
    input_data  = [9.099999609651045, 2.1549919990126183, 1.5752182041381115, 21.391284427002535, 0.9101106864202795,
                  0.8802054162523129, 0.8971994049002746, 0.8969494640950004, 0.8987228185093691, 0.9306593868837868,
                  0.9450679400258979]

    # new engine model validation
    # input_data  = [9.099579445479264, 2.1915909089517114, 1.5610148690902443, 20.95390258937267, 0.9155096296918898,
    #                0.8874570146206988, 0.8944829465164816, 0.9075182101371015, 0.9074727286671787, 0.9376350508985418,
    #                0.929797993929104]
    # input_data  = [9.099999609651045, 2.1549919990126183, 1.5752182041381115, 21.391284427002535, 0.8688,
    #               0.8667, 0.8971994049002746, 0.8969494640950004, 0.8920, 0.9306593868837868,
    #               0.9450679400258979]


GSPfileName = "DP2_new.mxl" if Engine == 0 else "GEnx-1B_V4DP_new.mxl"
# load the model
gspdll = loadModel(Engine, GSPfileName)

# run GSP
y_sim = runGsp(gspdll, input_data, output_list)[0]

# true values
y_true = true_val_DP_CF6 if Engine == 0 else true_val_DP_GEnx

outp = y_sim

# %% define constraints
def constr_f(x):

    return np.array(x[inputs_list.index('HPC_ef')] - etaP_C(
        loading(p.inputs['N2'], p.stagesHPC, p.RmHPC, outp[output_list.index("dH_HPC")])))


def constr_f2(x):

    return np.array(x[inputs_list.index("FPRbp")] - FPR(outp[output_list.index("Eta_fan")],
                                                        x[inputs_list.index('LPT_ef')], x[inputs_list.index('BPR')],
                                                        outp[output_list.index("FN")] * 1000,
                                                        p.inputs['dot_w'], p.inputs['Ta'], p.inputs['Ma']))

def constr_f3(x):

    return np.array(p.Abp - outp[output_list.index("A_bp")])

def constr_f4(x):

    return np.array(p.Ac - outp[output_list.index("A_c")])

def constr_f5(x):

    return np.array(x[inputs_list.index('fan_efC')] - etaP_C(loading(p.inputs['N1'], p.stagesB, p.RmfanC,
                                                                          outp[output_list.index(
                                                                              "dH_FanC")]))) #- 0.035 #TODO addded for the GENX

def constr_f6(x):

    return np.array(x[inputs_list.index('fan_efD')] - etaP_f(loading(p.inputs['N1'], 1, p.RmfanBp,
                                                                          outp[output_list.index("dH_FanBp")])))

def constr_f7(x):
    PrFcore = p.inputs['PRFcore'] if Engine == 0 else x[inputs_list.index("FPRc")]
    return PrFcore * x[inputs_list.index("HPCpr")] - p.OPR

def constr_f8(x):

    flowCin = flow_coef(p.inputs['N2'], p.RmHPTin, outp[output_list.index("VxHPTin")])
    flowCout = flow_coef(p.inputs['N2'], p.RmHPTout, outp[output_list.index("VxHPTout")])
    flowC = 0.5 * (flowCout + flowCin)
    lding = loading(p.inputs['N2'], p.stagesHPT, 0.5 * (p.RmHPTin + p.RmHPTout), -outp[output_list.index("dH_HPT")])

    return np.array(x[inputs_list.index('HPT_ef')] - rbf(flowC, lding))


def constr_f9(x):

    V_LPT = 0.5 * (outp[output_list.index("VxHPTout")] + outp[output_list.index("VxLPTout")])
    flowC = flow_coef(p.inputs['N1'], p.RmLPT, V_LPT)

    lding = loading(p.inputs['N1'], p.stagesLPT, p.RmLPT, -outp[output_list.index("dH_LPT")])

    return np.array(x[inputs_list.index('LPT_ef')] - rbf(flowC, lding))

print("RMS: ", np.sqrt(np.mean(((y_true - y_sim[:len(y_true)]) / (y_true + 0.000001)) ** 2)))
# print("C1: ", constr_f(input_data))
# print("C2: ", constr_f2(input_data))
# print("C3: ", constr_f3(input_data))
# print("C4: ", constr_f4(input_data))
# print("C5: ", constr_f5(input_data))
# print("C6: ", constr_f6(input_data))
# print("C7: ", constr_f7(input_data))
# print("C8: ", constr_f8(input_data))
# print("C9: ", constr_f9(input_data))
# %%
y_sim = y_sim[:len(y_true)]
errors = 100 * (y_true - y_sim[:len(y_true)]) / (y_true + 0.000001)
errorsdev = errors
ticks = output_list[:len(y_true)]
ticksdev = ticks
plt.scatter(ticks, errors, c='orange', marker="o", edgecolors='orange', s=80)
plt.grid()
plt.xlabel("Parameter")
plt.ylabel("Error (true-sim)[%]")
# plt.savefig('CF6_new.png', dpi=300)
plt.show()
#%%
inputs_list = ["BPR", "FPRc", "FPRbp", "HPCpr", 'HPC_ef', 'fan_efC', 'HPT_ef', 'LPT_ef', 'fan_efD', 'Cx_core', 'Cx_bp']
ticks = inputs_list
if Engine == 0:
    input_data  = [5.084086470704756, PRFanc, 1.7280973721923745, 12.97756833279673, 0.8973505330918321, 0.903470456888131,
             0.9082592265792493, 0.8719695048669192, 0.9181738420687525, 0.9188437011863649, 0.9270913176702706]

errors = 100 * (input_data - old_modelIn) / (old_modelIn + 0.000001)
plt.figure(figsize=(10, 4), dpi=100)
plt.scatter(ticks, errors, c='blue', marker="o", edgecolors='blue', s=80)
plt.grid()
plt.xlabel("Parameter")
plt.ylabel("Difference (new - old) [%]")
plt.show()

#%%
def barC(outputval, params_out, y_name):
    plt.rcParams['figure.dpi'] = 300
    outp_length = len(outputval)
    len_k       = 1
    w = 5  # in inch the width of the figure was 6
    h = 3  # height in inch was 3
    r = np.arange(outp_length)
    r = r * w / outp_length
    width = 0.25  # the bar with was0.18
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    colorl  = ['orange']

    i=0
    rec = ax.bar(r + width*i + 0.08, outputval, color=colorl[i], width=width, edgecolor=colorl[i],
           tick_label=outputval[i])
        # ax.bar_label(rec, padding=3)
    ax.yaxis.grid()  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest
    plt.xlabel("Parameters", fontsize=11)
    plt.ylabel(y_name, fontsize=11)

    plt.ylim(-0.3, 0.5)
    plt.xticks(r + width * len_k / 3, params_out)  # was 2.6
    # plt.legend(loc='upper left') # lower
    fig.tight_layout()
    plt.margins(y=0.1)
    plt.show()

barC(errorsdev, ticksdev, "Error (true-sim)[%]")


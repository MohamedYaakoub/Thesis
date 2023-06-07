"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 6/10/2022 9:55 AM  
"""
import pickle
import numpy as np
import timeit
import sys
from GSP_helper import cleanup, runGsp
import subprocess

sys.path.insert(1, "C:/Users/Shivan/OneDrive - Delft University of Technology/Desktop/Docs/VM/Parallel GSP/Shared "
                   "Folder/GSP")
from matplotlib import pyplot as plt
from parameters import params, params2, CF6_OD, CF6_OD_true, N2p_list, GEnx_OD_true, GEnx_OD, N2p_listgx

# %%
Engine = 0  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesign.mxl" if Engine == 0 else "OffDesignGEnx.mxl"
trueVal = CF6_OD_true if Engine == 0 else GEnx_OD_true
inputDat = CF6_OD if Engine == 0 else GEnx_OD
# note: these should be identical to the parameters defined in the API module of GSP
if Engine == 0:
    inputs_list = ["N1", "P0", "T0", "Rhum"]
    output_list = ["TT25", "Pt25", "Ps14", "TT3", "Ps3", "Pt49", "TT49", "T5", "FN", "W2", "Wf", "N2"]

else:
    inputs_list = ["N1", "P0", "T0", "Rhum"]
    output_list = ["TT25", "TT3", "Ps3", "TT49", "FN", "W2", "Wf", "N2"]

# dump the following to transfer them to the objective function file
pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))
# import the objective function
from OD_function import objScale, gspdll, reset_maps
# %%
# objScale([0.12866807435644212, 0.05396637713061947, 0.7417043096320264, 0.023033582019752474, 0.9843049967383277, -0.19266098050812622, 0.042378075957044015, 0.057986609188292085, 0.007331490333573276, -0.08322898354882793, 0.710007276958925, 0.2439733790396464, 0.2624643202299448, -0.012836134155575163, 0.9990041362353956, -0.045639047383945586, -0.1216219239636227, -0.03967984011022662, -0.0495492427344264, 0.019663828150158703, -0.47668924862872764, -0.038016589403356275]
#            )
# %%
start        = timeit.default_timer()
y_sim_new    = np.array(runGsp(gspdll, inputDat, output_list))
# N2p_simN     = y_sim_new[:, -1]
# y_sim_new    = y_sim_new[:, :-1]
reset_maps()
gspdll.InitializeModel()
y_sim_old    = np.array(runGsp(gspdll, inputDat, output_list))
# N2p_simO     = y_sim_old[:, -1]
# y_sim_old    = y_sim_old[:, :-1]
change       = np.mean((y_sim_new-y_sim_old)/(y_sim_old+0.000001), axis=0)
mean_change  = 100*np.sqrt(change**2)
end          = timeit.default_timer()
print("Time: ", end - start)
# %%
change_old   = (trueVal-y_sim_old)/(trueVal+0.000001)
change_new   = (trueVal-y_sim_new)/(trueVal+0.000001)
mean_old     = 100*np.sqrt(np.mean(change_old**2, axis=0))
mean_new     = 100*np.sqrt(np.mean(change_new**2, axis=0))
print(np.mean(mean_old), np.mean(mean_new))
# %%
# output_list = output_list[:-1]
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(output_list, mean_old, c='blue', marker="o", edgecolors='blue', s=80, label='before adaption')
ax1.scatter(output_list, mean_new, c='orange', marker="o", edgecolors='orange', s=80, label='after adaption')
ax1.grid()
ax1.set_xlabel("Parameter")
ax1.set_ylabel("mean error (true-sim) [%]")
ax1.legend()

ax2.scatter(output_list, mean_change, c='blue', marker="o", edgecolors='blue', s=80)
ax2.grid()
ax2.set_xlabel("Parameter")
ax2.set_ylabel("mean change (old-new) [%]")
plt.tight_layout()
plt.show()
# %%
cmap  = plt.get_cmap('tab20')
clist = cmap(np.linspace(0, 1, len(output_list)))
change_old   = 100*(trueVal-y_sim_old)/(trueVal+0.000001)
change_new   = 100*(trueVal-y_sim_new)/(trueVal+0.000001)
plt.figure()
for i in range(len(output_list)):
    plt.plot(inputDat[:, 0], change_old[:, i], c=clist[i], linestyle='-', label=output_list[i])
    plt.plot(inputDat[:, 0], change_new[:, i], c=clist[i], linestyle='--', label=output_list[i])
plt.xlabel('N1')
plt.ylabel('Error [%]')
plt.legend(loc='upper right')
plt.show()
# %%
cleanup(gspdll)
# %%
def barC(outputval, selected_k, params_out, y_name):
    plt.rcParams['figure.dpi'] = 300
    outp_length = len(outputval[0])
    len_k       = len(selected_k)
    w = 9  # in inch the width of the figure was 6
    h = 4  # height in inch was 3
    r = np.arange(outp_length)
    r = r * w / outp_length
    width = 0.25  # the bar with was0.18
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    colorl  = ['blue', 'orange']

    for i in range(len_k):
        label      = selected_k[i]
        rec = ax.bar(r + width*i, np.round(outputval[i], 1), color=colorl[i], width=width, edgecolor=colorl[i], label=label,
               tick_label=outputval[i])
        # ax.bar_label(rec, padding=3)
    ax.yaxis.grid()  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest
    plt.xlabel("Parameters", fontsize=14)
    plt.ylabel(y_name, fontsize=14)
    # plt.title("Sensitivity ")
    plt.ylim(0, 4.5)
    plt.xticks(r + width * len_k / 4, params_out)  # was 2.6
    plt.legend(loc='upper left') # lower
    fig.tight_layout()
    plt.margins(y=0.1)
    plt.show()

barC([list(mean_old.T), list(mean_new.T)], ["before adaption", "after adaption"], output_list, "Mean change (old-new) [%]")


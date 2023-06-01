"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 5/23/2022 11:41 AM  
"""
from scipy.optimize import differential_evolution, NonlinearConstraint
import pickle
import numpy as np
import timeit
import sys
from GSP_helper import cleanup, runGsp
import subprocess
sys.path.insert(1, "C:/Users/mohsy/University/KLM/Thesis/My thesis/Parallel GSP/Shared "
                   "Folder/GSP")
from matplotlib import pyplot as plt
from parameters import params, params2, CF6_OD, CF6_OD_true, GEnx_OD_true, GEnx_OD

# select the engine: 0 for the CF6, and 1 for the GEnx
Engine = 1
# set the GSP file names
GSPfileName = "OffDesign.mxl" if Engine == 0 else "OffDesignGEnx.mxl"
# all the parameters
p = params() if Engine == 0 else params2()
inputDat = CF6_OD if Engine == 0 else GEnx_OD

# %% Specify inputs for the Ga function
iters  = 20  # the number of iterations or also known as generations
pop    = 7   # the population size for each generation
tol    = 0.0001  # the tolerance value for termination
Nfeval = 1  # iteration number
# %% specify inputs and outputs for GSP
# note: these should be identical to the parameters defined in the API module of GSP
if Engine == 0:
    inputs_list = ["N1", "P0", "T0", "Rhum"]
    output_list = ["TT25", "Pt25", "Ps14", "TT3", "Ps3", "Pt49", "TT49", "T5", "FN", "W2", "Wf", "N2"]
    # order is : lin  - q or b - c
    # params are: PR, Mdot, Eta
    bounds = [(-0.7, 0.7),   (-0.1, 0.1),   (-0.5, 0.5), (-0.1, 0.1), (0.01, 0.2),  (-0.1, -0.01),  # fanC bounds
              (-1, 1),       (-0.1, 0.1),   (-1, 1),     (-0.1, 0.1), (-0.04, 0.2), (-0.1, 0.05),   # fanB bounds
              (-0.7, 0.7),   (-0.1, 0.1),   (-0.7, 0.7), (-0.1, 0.1), (-1, 1),      (-0.25, 0.25),  # HPC bounds
              (-.25, .25),   (-0.1, 0.1),  # HPT bounds
              (-.25, .25),   (-0.1, 0.1)]  # LPT bounds

    # bounds = [(-0.7, 0.7),   (-0.1, 0.1),   (-0.5, 0.5), (-0.1, 0.1), (0.02, 0.2),  (-0.1, -0.02),  # fanC bounds
    #           (-1, 1),       (-0.1, 0.1),   (-1, 1),     (-0.1, 0.1), (-0.04, 0.2), (-0.1, 0.05),   # fanB bounds
    #           (-0.7, 0.7),   (-0.1, 0.1),   (-0.7, 0.7), (-0.1, 0.1), (-1, 1),      (-0.25, 0.25),  # HPC bounds
    #           (-.25, .25),   (-0.1, 0.1),  # HPT bounds
    #           (-.25, .25),   (-0.1, 0.1)]  # LPT bounds


else:
    inputs_list = ["N1", "P0", "T0", "Rhum"]
    output_list = ["TT25", "TT3", "Ps3", "TT49", "FN", "W2", "Wf", "N2"]
    # order is : lin  - q or b - c
    # params are: PR, Mdot, Eta
    bounds = [(-1, 1),   (-0.1, 0.1),   (-1, 1), (-0.1, 0.1), (-0.2, 0.2), (-0.15, 0.15),  # fanC bounds
              (-1, 1),   (-0.1, 0.1),   (-1, 1), (-0.1, 0.1), (-0.2, 0.2), (-0.15, 0.15),  # fanB bounds
              (-1, 1),   (-0.1, 0.1),   (-1, 1), (-0.1, 0.1), (-0.1, 0.2), (-0.15, 0.15),    # HPC bounds
              (-.25, .25),   (-0.1, 0.1),  # HPT bounds
              (-.25, .25),   (-0.1, 0.1)]  # LPT bounds

# dump the following to transfer them to the objective function file
pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))
# import the objective function
from OD_function import objFOD, gspdll, scaling_F, reset_maps, plot_maps, plot_SF

# reset the modified map to the reference map
reset_maps()

iter_Xi     = []  # list with the fittest individual of each generation
iter_objfun = []  # list with objective function values
iter_time   = []  # list containing the duration for each iteration
# %% progress bar plotting function

def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '_' * (bar_len - filled_len)

    sys.stdout.write('\r%s' '/' '%s %s %s%s %s' % (count, total, bar, percents, '%', status))
    sys.stdout.flush()

# %% A function to follow the progress of the minimization.
def callbackF(Xi, convergence):  # only 1 input variable for scipy.optimize.minimize
    global Nfeval
    global iters
    time = timeit.default_timer()
    iter_Xi.append(Xi)
    iter_objfun.append(objFOD(Xi))
    iter_time.append(time)
    status = "GA is running..."
    if Nfeval == iters:
        status = "GA finished"
    progress(Nfeval, iters, status=status)
    Nfeval += 1

def constr_f(x):
    return x[0] - x[1]

nlc = NonlinearConstraint(constr_f, -5, 5)

# %% run GA
start = timeit.default_timer()  # initiate time
iter_time.append(start)
result = differential_evolution(objFOD,
                                bounds,
                                strategy='best1bin',
                                popsize=pop,
                                maxiter=iters,
                                tol=tol,
                                polish=False,
                                # constraints= (nlc),
                                # x0=x0,
                                callback=callbackF,
                                mutation=[0, 1],   # todo: changed
                                seed=5325,  # 5325
                                recombination=0.7)
# %% print and plot the results
end = timeit.default_timer()  # end time
y_sim = np.array(runGsp(gspdll, inputDat, output_list))
y_true = CF6_OD_true if Engine == 0 else GEnx_OD_true
# %%
Rms = np.sqrt(np.mean(((y_true - y_sim) / (y_true + 0.000001)) ** 2, axis=0))
error_mat = (y_true-y_sim)/y_true

print('\n %s %s' % ("      Objective:", result['fun']))
print('%s %s' % ("Design variables:", list(result['x'])))
print('            Time:', end - start)  # optimisation time
print('             RMS:', Rms)
# plot convergence
plt.plot(iter_objfun)
plt.xlabel('Iteration')
plt.ylabel('Objective function')
plt.title('Genetic Algorithm')
plt.show()
# plot the time for each iter
plt.scatter(range(1, len(iter_time)), [(iter_time[i+1]-iter_time[i])/60 for i in range(len(iter_time)-1)])
plt.xlabel('Iteration')
plt.ylabel('Time [min]')
plt.title('Genetic Algorithm time')
plt.show()

# %% plot the resulting and starting maps

plot_maps('C', "1_LPC_core")
plot_maps('C', "2_LPC_bypass")
plot_maps('C', "3_HPC")
plot_maps('T', "4_HPT")
plot_maps('T', "5_LPT")

# %% plot the scaling factors
plot_SF(1, 'C',  "1_LPC_core", result['x'][:6])
plot_SF(1, 'C', "2_LPC_bypass", result['x'][6: 12])
plot_SF(2, 'C', "3_HPC", result['x'][12: 18])
plot_SF(2, 'T', "4_HPT", result['x'][18:20])
plot_SF(1, 'T', "5_LPT", result['x'][20:22])
# %%
print("1_LPC_core   :", result['x'][:6])
print("2_LPC_bypass :", result['x'][6: 12])
print("3_HPC        :", result['x'][12: 18])
print("4_HPT        :", result['x'][18:20])
print("5_LPT        :", result['x'][20:22])

# %%
cleanup(gspdll)

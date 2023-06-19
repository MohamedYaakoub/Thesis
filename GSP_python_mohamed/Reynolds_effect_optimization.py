import scipy
from scipy.optimize import differential_evolution, NonlinearConstraint
import pickle
import numpy as np
import timeit
import sys
from GSP_helper import cleanup, runGsp
from matplotlib import pyplot as plt

from Reynolds_effect_functions import*
from map_functions import reset_maps, plot_maps

iters  = 20  # the number of iterations or also known as generations
pop    = 4 # the population size for each generation
tol    = 0.0001  # the tolerance value for termination
# tol    = 10 # the tolerance value for termination
Nfeval = 1  # iteration number

# bounds = [(-1, 1),   (-0.1, 0.1),   (-1, 1), (-0.1, 0.1), (-0.2, 0.2), (-0.15, 0.15),  # fanC bounds
#               (-1, 1),   (-0.1, 0.1),   (-1, 1), (-0.1, 0.1), (-0.2, 0.2), (-0.15, 0.15),  # fanB bounds
#               (-1, 1),   (-0.1, 0.1),   (-1, 1), (-0.1, 0.1), (-0.1, 0.2), (-0.15, 0.15),    # HPC bounds
#               (-.25, .25),   (-0.1, 0.1),  # HPT bounds
#               (-.25, .25),   (-0.1, 0.1)]  # LPT bounds
# x0 = [-0.1406342786398378, 0.03998946680240654, -0.019627303516216554,
#       -0.07005805929844654, 0.05494991400449703, 0.0014290219083181288,
#       -0.006148718993880364, -0.03026666562309781, -0.37485030931945695,
#       -0.023378337504179204, 0.06558515042176159, -0.026524631545875397,
#       -0.3358971894455346, 0.039868089034289095, 0.07724064473013237,
#       0.09999623027761487, 0.07314566414903378, -0.11496762120404547,
#       0.24714318479666414, -0.08900711137883877, 0.11565454431068517,
#       -0.0844351677855871]
# x0 = np.zeros((22,1))
bounds = 22 * [(-1,1)]
x0 = [-0.99820857, 0.00206351, 0.32278815, -0.04916206, -0.06706796, -0.02361838,
       0.17766424, -0.07706789, -0.55033663, 0.02106336, -0.12588236, 0.06232044,
       -0.20784916, -0.00502733, -0.45136154, 0.04389881, 0.006958, 0.04694079,
       -0.2439229, 0.05428811, -0.03710134, -0.00842078]

reset_maps()

iter_Xi     = []  # list with the fittest individual of each generation
iter_objfun = []  # list with objective function values
iter_time   = []  # list containing the duration for each iteration




def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '_' * (bar_len - filled_len)

    sys.stdout.write('\r%s' '/' '%s %s %s%s %s' % (count, total, bar, percents, '%', status))
    sys.stdout.flush()

def callbackF(Xi, convergence):  # only 1 input variable for scipy.optimize.minimize
# def callbackF(Xi):  # only 1 input variable for scipy.optimize.minimize
    global Nfeval
    global iters
    print(Xi, "iters")
    time = timeit.default_timer()
    iter_Xi.append(Xi)
    # iter_objfun.append(objFOD(Xi))
    iter_time.append(time)
    status = "GA is running..."
    if Nfeval == iters:
        status = "GA finished"
    progress(Nfeval, iters, status=status)
    Nfeval += 1



start = timeit.default_timer()  # initiate time
iter_time.append(start)
result = differential_evolution(objFOD,
                                bounds,
                                strategy='best1bin',
                                popsize=pop,
                                maxiter=iters,
                                tol=tol,
                                x0=x0,
                                polish=False,
                                # constraints= (nlc),
                                # x0=x0,
                                # callback=callbackF,
                                disp=True,
                                callback=callbackF,
                                mutation=[0, 1],   # todo: changed
                                seed=5325,  # 5325
                                recombination=0.7)

# result = scipy.optimize.minimize(objFOD, x0, method='SLSQP',
#                                  jac=None, hess=None, hessp=None,
#                                  bounds=bounds, tol=tol,
#                                  callback=callbackF, options={'disp': True})




end = timeit.default_timer()  # end time
y_sim = np.array(runGsp(gspdll, GEnx_OD, output_list))[:, :6]
y_true = GEnx_OD_true
# %%
Rms = np.sqrt(np.mean(((y_true - y_sim) / (y_true + 0.000001)) ** 2, axis=0))
error_mat = (y_true-y_sim)/y_true
print(result)
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
# plot_SF(1, 'C',  "1_LPC_core", result['x'][:6])
# plot_SF(1, 'C', "2_LPC_bypass", result['x'][6: 12])
# plot_SF(2, 'C', "3_HPC", result['x'][12: 18])
# plot_SF(2, 'T', "4_HPT", result['x'][18:20])
# plot_SF(1, 'T', "5_LPT", result['x'][20:22])
# %%
print("1_LPC_core   :", result['x'][:6])
print("2_LPC_bypass :", result['x'][6: 12])
print("3_HPC        :", result['x'][12: 18])
print("4_HPT        :", result['x'][18:20])
print("5_LPT        :", result['x'][20:22])

# %%
cleanup(gspdll)

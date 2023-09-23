import scipy
from scipy.optimize import differential_evolution, NonlinearConstraint
import pickle
import numpy as np
import timeit
import sys
from GSP_helper import cleanup, runGsp
from matplotlib import pyplot as plt

from Reynolds_effect_functions import *
from map_functions import reset_maps, plot_maps

iters = 50  # the number of iterations or also known as generations
pop = 4  # the population size for each generation
tol = 0.0001  # the tolerance value for termination
Nfeval = 1  # iteration number

# bounds = [(-0.1, 0.1),   (-0.25, 0.1),   (-0.1, 0.1),   (-0.25, 0.1), (-0.1, 0.1),   (-0.25, 0.1),  # fanC bounds
#               (-0.1, 0.1),   (-0.2, 0.05),   (-0.1, 0.1),   (-0.2, 0.05), (-0.1, 0.1),   (-0.2, 0.05),  # fanB bounds
#               (-0.1, 0.1),   (-0.2, 0.05),   (-0.1, 0.1),   (-0.2, 0.05), (-0.1, 0.1),   (-0.2, 0.05),    # HPC bounds
#               (-0.1, 0.1),   (-0.2, 0.05),  # HPT bounds
#               (-0.1, 0.1),   (-0.2, 0.05)]  # LPT bounds

bounds = [(-0.1, 0.1), (-0.25, 0.1), (-0.1, 0.1), (-0.25, 0.1),  # fanC bounds
          (-0.1, 0.1), (-0.25, 0.1), (-0.1, 0.1), (-0.25, 0.1),  # fanB bounds
          (-0.1, 0.1), (-0.2, 0.1), (-0.1, 0.1), (-0.2, 0.1),  # HPC bounds
          (-0.1, 0.1), (-0.2, 0.1),  # HPT bounds
          (-0.1, 0.1), (-0.2, 0.1)]  # LPT bounds

# bounds = 22 * [(-1,1)]


# x0 = [0] * 16
x0 = [-0.04371352, 0.0813069, -0.09591862, 0.08377887, -0.07021975,
      -0.0691131, -0.05098189, 0.09924948, 0.04400945, -0.19977984,
      0.03704963, 0.00522859, -0.00374285, 0.02801069, -0.03519336,
      -0.06247652]

# reset_maps()

iter_Xi = []  # list with the fittest individual of each generation
iter_objfun = []  # list with objective function values
iter_time = []  # list containing the duration for each iteration


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
    file = open('New solves OD scaling/Running solve.txt', 'a')
    file.write(str(Xi) + " iters\n")
    file.close()
    time = timeit.default_timer()
    iter_Xi.append(Xi)
    # iter_objfun.append(objFOD(Xi))
    iter_time.append(time)

    pickle.dump([iter_time, iter_Xi], open("New solves OD scaling/solver output.p", "wb"))
    status = "GA is running..."
    if Nfeval == iters:
        status = "GA finished"
    progress(Nfeval, iters, status=status)
    Nfeval += 1


# constraints = ({'type': 'eq', 'fun': equality_constraint_cruise_climb_smoothness})


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
                                mutation=[0, 1],  # todo: changed
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
error_mat = (y_true - y_sim) / y_true
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
plt.scatter(range(1, len(iter_time)), [(iter_time[i + 1] - iter_time[i]) / 60 for i in range(len(iter_time) - 1)])
plt.xlabel('Iteration')
plt.ylabel('Time [min]')
plt.title('Genetic Algorithm time')
plt.show()

# %% plot the resulting and starting maps

# plot_maps('C', "1_LPC_core")
# plot_maps('C', "2_LPC_bypass")
# plot_maps('C', "3_HPC")
# plot_maps('T', "4_HPT")
# plot_maps('T', "5_LPT")

# %% plot the scaling factors
# plot_SF(1, 'C',  "1_LPC_core", result['x'][:6])
# plot_SF(1, 'C', "2_LPC_bypass", result['x'][6: 12])
# plot_SF(2, 'C', "3_HPC", result['x'][12: 18])
# plot_SF(2, 'T', "4_HPT", result['x'][18:20])
# plot_SF(1, 'T', "5_LPT", result['x'][20:22])
# %%
# print("1_LPC_core   :", result['x'][:6])
# print("2_LPC_bypass :", result['x'][6: 12])
# print("3_HPC        :", result['x'][12: 18])
# print("4_HPT        :", result['x'][18:20])
# print("5_LPT        :", result['x'][20:22])

# %%
cleanup(gspdll)

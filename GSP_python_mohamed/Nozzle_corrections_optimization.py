from scipy.optimize import differential_evolution, NonlinearConstraint
import timeit
import sys
from GSP_helper import cleanup, runGsp
from matplotlib import pyplot as plt
from Nozzle_corrections_functions import *

iters = 300  # the number of iterations or also known as generations
pop = 4  # the population size for each generation
tol = 0.0001  # the tolerance value for termination
Nfeval = 1  # iteration number

# bounds = [(-0.1, 0.1),   (-0.25, 0.1),   (-0.1, 0.1),   (-0.25, 0.1), (-0.1, 0.1),   (-0.25, 0.1),  # fanC bounds
#               (-0.1, 0.1),   (-0.2, 0.05),   (-0.1, 0.1),   (-0.2, 0.05), (-0.1, 0.1),   (-0.2, 0.05),  # fanB bounds
#               (-0.1, 0.1),   (-0.2, 0.05),   (-0.1, 0.1),   (-0.2, 0.05), (-0.1, 0.1),   (-0.2, 0.05),    # HPC bounds
#               (-0.1, 0.1),   (-0.2, 0.05),  # HPT bounds
#               (-0.1, 0.1),   (-0.2, 0.05)]  # LPT bounds

bounds = [(-0.1, 0.1), (-0.2, 0.2), (-0.1, 0.1), (-0.2, 0.2),  # fanC bounds
          (-0.1, 0.1), (-0.2, 0.1), (-0.1, 0.1), (-0.2, 0.1),  # fanB bounds
          (-0.1, 0.1), (-0.2, 0.1), (-0.1, 0.1), (-0.2, 0.1),  # HPC bounds
          (-0.1, 0.1), (-0.2, 0.1),  # HPT bounds
          (-0.1, 0.1), (-0.2, 0.1),  # LPT bounds
          (-0.2, 0.0), (-0.08, 0.16),  # core nozzle-0.2, 0.0, -0.08, 0.16
          (-0.1, 0.0), (-0.08, 0.08)]  # bypass nozzle

x0 = [-0.09737036, 0.09429092, -0.09994714, 0.19372794, -0.02110662, 0.09311127,
      -0.08758217, 0.04846549, 0.05750697, -0.18643443, 0.01754303, -0.08327033,
      -0.04006111, 0.02969302, -0.02456914, -0.06138784, -0.09979278, 0.07921487,
      -0.04021015, 0.0657303]

# x0 = [-0.03794534, -0.19971319, -0.09351238, 0.19967979, 0.06058644,
#       0.05116728, 0.04537528, 0.09770786, -0.05279553, 0.03111123,
#       0.01554495, -0.13347183, -0.08563969, -0.00759692, 0.00394694, 0.09293318,
#       0, 0, 0, 0,
#       0, 0, 0, 0]

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

    pickle.dump([iter_time, iter_Xi], open("New solves OD scaling/Nozzle single equation iter4 - only Cv.p", "wb"))
    status = "GA is running..."
    if Nfeval == iters:
        status = "GA finished"
    progress(Nfeval, iters, status=status)
    Nfeval += 1


# from splines_smoothness_constraints import \
#     equality_Fan_c_fm_cruise_climb, equality_Fan_c_fe_cruise_climb, \
#     equality_Fan_d_fm_cruise_climb, equality_Fan_d_fe_cruise_climb, \
#     equality_HPC_fm_cruise_climb, equality_HPC_fe_cruise_climb, \
#     equality_HPT_fe_cruise_climb, \
#     equality_LPT_fe_cruise_climb, \
#     equality_Fan_c_fm_der_cruise_climb, equality_Fan_c_fe_der_cruise_climb, \
#     equality_Fan_d_fm_der_cruise_climb, equality_Fan_d_fe_der_cruise_climb, \
#     equality_HPC_fm_der_cruise_climb, equality_HPC_fe_der_cruise_climb, \
#     equality_HPT_fe_der_cruise_climb, \
#     equality_LPT_fe_der_cruise_climb
#
# limit = 0.005
# limit_der = 2e-07
#
# equality_constraints = (
#     NonlinearConstraint(equality_Fan_c_fm_cruise_climb, -limit, limit),
#     NonlinearConstraint(equality_Fan_c_fe_cruise_climb, -limit, limit),
#     NonlinearConstraint(equality_Fan_d_fm_cruise_climb, -limit, limit),
#     NonlinearConstraint(equality_Fan_d_fe_cruise_climb, -limit, limit),
#     NonlinearConstraint(equality_HPC_fm_cruise_climb, -limit, limit),
#     NonlinearConstraint(equality_HPC_fe_cruise_climb, -limit, limit),
#     NonlinearConstraint(equality_HPT_fe_cruise_climb, -limit, limit),
#     NonlinearConstraint(equality_LPT_fe_cruise_climb, -limit, limit),
#     NonlinearConstraint(equality_Fan_c_fm_der_cruise_climb, -limit_der, limit_der),  # der start
#     NonlinearConstraint(equality_Fan_c_fe_der_cruise_climb, -limit_der, limit_der),
#     NonlinearConstraint(equality_Fan_d_fm_der_cruise_climb, -limit_der, limit_der),
#     NonlinearConstraint(equality_Fan_d_fe_der_cruise_climb, -limit_der, limit_der),
#     NonlinearConstraint(equality_HPC_fm_der_cruise_climb, -limit_der, limit_der),
#     NonlinearConstraint(equality_HPC_fe_der_cruise_climb, -limit_der, limit_der),
#     NonlinearConstraint(equality_HPT_fe_der_cruise_climb, -limit_der, limit_der),
#     NonlinearConstraint(equality_LPT_fe_der_cruise_climb, -limit_der, limit_der)
# )

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
                                # constraints=equality_constraints,
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

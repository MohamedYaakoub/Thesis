from scipy.optimize import differential_evolution, NonlinearConstraint
import timeit
import sys
from GSP_helper import cleanup, runGsp
from matplotlib import pyplot as plt
from Reynolds_effect_functions import *

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
          (-0.1, 0.1), (-0.2, 0.1)]  # LPT bounds

x0 = [-0.02223825, 0.09164879, -0.09991125, 0.19766541, -0.00923643, 0.0319769,
      - 0.09198503, 0.0346309, 0.04168849, -0.19995974, 0.01842454, -0.09209377,
      - 0.04210804, 0.0318016, -0.09989942, -0.10127006]

# x0 = [-0.035561886402363796, 0.03731023498770241, 0.08991042836121235,
#                    0.19880644744872245, -0.0754123113935967, -0.01267174391720622,
#                    -0.040844520243467124, 0.02743580134539371, 0.02555529358963049,
#                    -0.12721143435875293, 0.005362532256188147, -0.021475633471666433,
#                    -0.027431178842507897, 0.052531285863499824, -0.09999874786120404,
#                    0.09999779570942106]

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

    pickle.dump([iter_time, iter_Xi], open("New solves OD scaling/Splines climb based on single eq.p", "wb"))
    status = "GA is running..."
    if Nfeval == iters:
        status = "GA finished"
    progress(Nfeval, iters, status=status)
    Nfeval += 1


from splines_smoothness_constraints import \
    equality_Fan_c_fm_cruise_climb, equality_Fan_c_fe_cruise_climb, \
    equality_Fan_d_fm_cruise_climb, equality_Fan_d_fe_cruise_climb, \
    equality_HPC_fm_cruise_climb, equality_HPC_fe_cruise_climb, \
    equality_HPT_fe_cruise_climb, \
    equality_LPT_fe_cruise_climb, \
    equality_Fan_c_fm_der_cruise_climb, equality_Fan_c_fe_der_cruise_climb, \
    equality_Fan_d_fm_der_cruise_climb, equality_Fan_d_fe_der_cruise_climb, \
    equality_HPC_fm_der_cruise_climb, equality_HPC_fe_der_cruise_climb, \
    equality_HPT_fe_der_cruise_climb, \
    equality_LPT_fe_der_cruise_climb

limit = 0.005
limit_der = 2e-07

equality_constraints = (
    NonlinearConstraint(equality_Fan_c_fm_cruise_climb, -limit, limit),
    NonlinearConstraint(equality_Fan_c_fe_cruise_climb, -limit, limit),
    NonlinearConstraint(equality_Fan_d_fm_cruise_climb, -limit, limit),
    NonlinearConstraint(equality_Fan_d_fe_cruise_climb, -limit, limit),
    NonlinearConstraint(equality_HPC_fm_cruise_climb, -limit, limit),
    NonlinearConstraint(equality_HPC_fe_cruise_climb, -limit, limit),
    NonlinearConstraint(equality_HPT_fe_cruise_climb, -limit, limit),
    NonlinearConstraint(equality_LPT_fe_cruise_climb, -limit, limit),
    NonlinearConstraint(equality_Fan_c_fm_der_cruise_climb, -limit_der, limit_der),  # der start
    NonlinearConstraint(equality_Fan_c_fe_der_cruise_climb, -limit_der, limit_der),
    NonlinearConstraint(equality_Fan_d_fm_der_cruise_climb, -limit_der, limit_der),
    NonlinearConstraint(equality_Fan_d_fe_der_cruise_climb, -limit_der, limit_der),
    NonlinearConstraint(equality_HPC_fm_der_cruise_climb, -limit_der, limit_der),
    NonlinearConstraint(equality_HPC_fe_der_cruise_climb, -limit_der, limit_der),
    NonlinearConstraint(equality_HPT_fe_der_cruise_climb, -limit_der, limit_der),
    NonlinearConstraint(equality_LPT_fe_der_cruise_climb, -limit_der, limit_der)
)

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
                                constraints=equality_constraints,
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

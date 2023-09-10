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
Nfeval = 1  # iteration number

# bounds = [(-1, 1),   (-0.1, 0.1),   (-1, 1), (-0.1, 0.1), (-0.2, 0.2), (-0.15, 0.15),  # fanC bounds
#               (-1, 1),   (-0.1, 0.1),   (-1, 1), (-0.1, 0.1), (-0.2, 0.2), (-0.15, 0.15),  # fanB bounds
#               (-1, 1),   (-0.1, 0.1),   (-1, 1), (-0.1, 0.1), (-0.1, 0.2), (-0.15, 0.15),    # HPC bounds
#               (-.25, .25),   (-0.1, 0.1),  # HPT bounds
#               (-.25, .25),   (-0.1, 0.1)]  # LPT bounds

bounds = 22 * [(-1,1)]

# X_takeoff = [-0.9983389133736509, 0.9385307347869887, 0.034000861580716135, -0.03577821327586628, -0.9981798259300767,
#              0.9530924154305069, -0.995700235073686, 0.8304600340231876, -0.13591022775450678, -0.11859373577835941,
#              -0.649524267286157, 0.5760310215827391, 0.09742118796651944, 0.1789460709533659, 0.04414420344277481,
#              0.1036752582677658, -0.5609002492275668, 0.8606842835909971, -0.423764318345767, 0.45272003883100775,
#              0.2975860084469224, -0.48489697839506674]

# X_climb = [-0.39151153304030584, -0.3572569487085121, -0.11508814965214664, 0.6040357595244259, -0.5473006373047256,
#            -0.4337901143312386, -0.9781608146532708, 0.9872048995550504, -0.9250162770160874, -0.3300894211317793,
#            -0.7196445323625377, -0.63761329701412, -0.37761050930438855, 0.37409468505190535, -0.7169121028867804,
#            0.504095168408869, 0.004084270145224167, 0.5007723795343721, -0.0414428972840476, 0.47249205746563394,
#            -0.7884995578550504, -0.2725844710128533]

X_cruise = [0.57412476, -0.90224694, 0.08398058, -0.30101288, -0.04481059, -0.10244869, 0.4126201, 0.26567765,
            -0.5879205, -0.29147754, 0.65966397, -0.28873865, -0.16545684, 0.63965032, -0.9634026, 0.65153249,
            -0.00253814, -0.84644459, -0.21551771, 0.20316737, -0.94047361, -0.05217237]

x0 = X_cruise

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

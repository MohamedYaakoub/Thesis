"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 8/19/2022 12:14 PM  
"""
from scipy.optimize import differential_evolution, NonlinearConstraint
import pickle
import numpy as np
import timeit
import sys
import subprocess

sys.path.insert(1, "C:/Users/Shivan/OneDrive - Delft University of Technology/Desktop/Docs/VM/Parallel GSP/Shared "
                   "Folder/GSP")
from matplotlib import pyplot as plt
from Fletcher_data import loading, etaP_f, etaP_C, flow_coef
from Optimum_param import FPR
from parameters import params, true_val_DP_CF6
from parameters_valid_testcell import true_val_DP_GEnx, params2
from smithChart import rbf

# select the engine: 0 for the CF6, and 1 for the GEnx
Engine = 1
# set the GSP file names
GSPfileName = "DP2_new.mxl" if Engine == 0 else "GEnx-1B_V4DP_new Valid.mxl"
trueVal = true_val_DP_CF6 if Engine == 0 else true_val_DP_GEnx
# all the parameters
p = params() if Engine == 0 else params2()

# %% specify inputs and outputs for GSP
# note: these should be identical to the parameters defined in the API module of GSP
if Engine == 0:
    inputs_list = ["BPR", "FPRbp", "HPCpr", 'HPC_ef', 'fan_etaC', 'HPT_ef', 'LPT_ef', 'fan_etaD', 'Cx_core', 'Cx_bp']
    output_list = ["TT25", "Ps14", "TT3", "Ps3", "Pt49", "TT49", "T5", "FN", "dH_HPC", "dH_FanC", "dH_FanBp",
                   "A_c", "A_bp", "Eta_fan", "VxHPTin", "VxHPTout", "VxLPTout", "dH_HPT", "dH_LPT"]

    bounds = [(5, 5.15), (1.5, 1.8), (10, 15),
              (0.85, 0.95), (0.85, 0.95), (0.85, 0.95), (0.85, 0.95), (0.85, 0.95), (0.90, 0.95),
              (0.90, 0.95)]  # bounds for the variables

else:
    inputs_list = ["BPR", "FPRc", "FPRbp", "HPCpr", 'HPC_ef', 'fan_etaC', 'HPT_ef', 'LPT_ef', 'fan_etaD', 'Cx_core',
                   'Cx_bp']
    output_list = ["TT25", "TT3", "Ps3", "TT49", "FN", "dH_HPC", "dH_FanC", "dH_FanBp",
                   "A_c", "A_bp", "Eta_fan", "VxHPTin", "VxHPTout", "VxLPTout", "dH_HPT", "dH_LPT"]

    bounds = [(8.7, 9.1), (2.1, 2.4), (1.4, 1.7), (19, 22),
              (0.85, 0.95), (0.88, 0.97), (0.85, 0.95), (0.85, 0.95), (0.88, 0.97), (0.90, 0.95),
              (0.90, 0.95)]  # bounds for the variables

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))
# import the objective function
from TestFile import objF, MultiGSP

for i in [[0.5, 1.5]]:  # 300 , [1, 1.5][0.0, 0.6]

    print("################### Run: " + str(i) + " #######################")

    # %% Specify inputs for the Ga function
    iters = 30  # the number of iterations or also known as generations
    pop = 15  # the population size for each generation
    tol = 0.0001  # the tolerance value for termination
    workers = 1  # amount of workers used for parallel computing

    x0 = [9.099999609651045, 2.1549919990126183, 1.5752182041381115, 21.391284427002535, 0.9101106864202795,
          0.8802054162523129, 0.8971994049002746, 0.8969494640950004, 0.8987228185093691, 0.9306593868837868,
          0.9450679400258979]

    # %%  create empty lists to collect the results
    F_eval = 0
    Nfeval = 1  # iteration number
    iter_Xi = []  # list with fittest individual of each individual
    iter_objfun = []  # list with objective function values


    # %% progress bar plotting function
    def progress(count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s' '/' '%s %s %s%s %s' % (count, total, bar, percents, '%', status))
        sys.stdout.flush()


    # %% define constraints
    def constr_f(x):

        global F_eval
        F_eval += 1
        outp = MultiGSP.run_simulations(x)[0]
        pickle.dump([outp, x], open("obj.p", "wb"))

        return np.array(x[inputs_list.index('HPC_ef')] - etaP_C(
            loading(p.inputs['N2'], p.stagesHPC, p.RmHPC, outp[output_list.index("dH_HPC")])))


    def constr_f2(x):
        outp, xp = pickle.load(open("obj.p", "rb"))
        if not set(x) == set(xp):
            global F_eval
            F_eval += 1
            outp = MultiGSP.run_simulations(x)[0]
        return np.array(x[inputs_list.index("FPRbp")] - FPR(outp[output_list.index("Eta_fan")],
                                                            x[inputs_list.index('LPT_ef')], x[inputs_list.index('BPR')],
                                                            outp[output_list.index("FN")] * 1000,
                                                            p.inputs['dot_w'], p.inputs['Ta'], p.inputs['Ma']))


    def constr_f3(x):
        outp, xp = pickle.load(open("obj.p", "rb"))
        if not set(x) == set(xp):
            global F_eval
            F_eval += 1
            outp = MultiGSP.run_simulations(x)[0]
        return np.array(p.Abp - outp[output_list.index("A_bp")])


    def constr_f4(x):
        outp, xp = pickle.load(open("obj.p", "rb"))
        if not set(x) == set(xp):
            global F_eval
            F_eval += 1
            outp = MultiGSP.run_simulations(x)[0]
        return np.array(p.Ac - outp[output_list.index("A_c")])


    def constr_f5(x):
        outp, xp = pickle.load(open("obj.p", "rb"))
        if not set(x) == set(xp):
            global F_eval
            F_eval += 1
            outp = MultiGSP.run_simulations(x)[0]
        return np.array(x[inputs_list.index('fan_etaC')] - etaP_C(loading(p.inputs['N1'], p.stagesB, p.RmfanC,
                                                                          outp[output_list.index(
                                                                              "dH_FanC")]))) - 0.035
                                                                            # TODO addded for the GENX
    def constr_f6(x):
        outp, xp = pickle.load(open("obj.p", "rb"))
        if not set(x) == set(xp):
            global F_eval
            F_eval += 1
            outp = MultiGSP.run_simulations(x)[0]
        return np.array(x[inputs_list.index('fan_etaD')] - etaP_f(loading(p.inputs['N1'], 1, p.RmfanBp,
                                                                          outp[output_list.index("dH_FanBp")])))


    def constr_f7(x):
        PrFcore = p.inputs['PRFcore'] if Engine == 0 else x[inputs_list.index("FPRc")]
        return p.OPR - PrFcore * x[inputs_list.index("HPCpr")]


    def constr_f8(x):
        outp, xp = pickle.load(open("obj.p", "rb"))
        if not set(x) == set(xp):
            global F_eval
            F_eval += 1
            outp = MultiGSP.run_simulations(x)[0]
        flowCin = flow_coef(p.inputs['N2'], p.RmHPTin, outp[output_list.index("VxHPTin")])
        flowCout = flow_coef(p.inputs['N2'], p.RmHPTout, outp[output_list.index("VxHPTout")])
        flowC = 0.5 * (flowCout + flowCin)
        lding = loading(p.inputs['N2'], p.stagesHPT, 0.5 * (p.RmHPTin + p.RmHPTout), -outp[output_list.index("dH_HPT")])

        return np.array(x[inputs_list.index('HPT_ef')] - rbf(flowC, lding))


    def constr_f9(x):
        outp, xp = pickle.load(open("obj.p", "rb"))
        if not set(x) == set(xp):
            global F_eval
            F_eval += 1
            outp = MultiGSP.run_simulations(x)[0]

        V_LPT = 0.5 * (outp[output_list.index("VxHPTout")] + outp[output_list.index("VxLPTout")])
        flowC = flow_coef(p.inputs['N1'], p.RmLPT, V_LPT)

        lding = loading(p.inputs['N1'], p.stagesLPT, p.RmLPT, -outp[output_list.index("dH_LPT")])

        return np.array(x[inputs_list.index('LPT_ef')] - rbf(flowC, lding))


    # use the non-lin constraint wrapper in order to make them compatible with the GA package

    if Engine == 0:
        nlc = NonlinearConstraint(constr_f, -0.0125, 0.0125)  # TODO changed

        nlc2 = NonlinearConstraint(constr_f2, -0.04, 0.04)

        nlc3 = NonlinearConstraint(constr_f3, -0.05, 0.05)

        nlc4 = NonlinearConstraint(constr_f4, -0.033, 0.03)  # TODO changed

        nlc5 = NonlinearConstraint(constr_f5, -0.01, 0.01)

        nlc6 = NonlinearConstraint(constr_f6, -0.01, 0.01)

        nlc7 = NonlinearConstraint(constr_f7, -1, 1.5)

        nlc8 = NonlinearConstraint(constr_f8, -0.025, 0.025)

        nlc9 = NonlinearConstraint(constr_f9, -0.025, 0.025)

    else:
        nlc = NonlinearConstraint(constr_f, -0.01, 0.03)

        nlc2 = NonlinearConstraint(constr_f2, -0.1, 0.1)

        nlc3 = NonlinearConstraint(constr_f3, -0.06, 0.06)

        nlc4 = NonlinearConstraint(constr_f4, -0.03, 0.03)

        nlc5 = NonlinearConstraint(constr_f5, -0.01, 0.01)  # todo increased

        nlc6 = NonlinearConstraint(constr_f6, -0.01, 0.02)

        nlc7 = NonlinearConstraint(constr_f7, -1, 1)

        nlc8 = NonlinearConstraint(constr_f8, -0.01, 0.01)

        nlc9 = NonlinearConstraint(constr_f9, -0.025, 0.025)  # todo increased


    # %% A function to follow the progress of the minimization.
    def callbackF(Xi, convergence):  # only 1 input variable for scipy.optimize.minimize
        global Nfeval
        global iters
        iter_Xi.append(Xi)
        iter_objfun.append(objF(Xi))
        status = "GA is running..."
        if Nfeval == iters:
            status = "GA finished"
        progress(Nfeval, iters, status=status)
        Nfeval += 1


    # %% GSP and objective function

    if __name__ == '__main__':

        # %% start optimisation and retrieve results

        start = timeit.default_timer()  # initiate time
        print("Starting optimisation...", end='\r')
        result = differential_evolution(objF, bounds,
                                        strategy='best1bin',
                                        popsize=pop,
                                        maxiter=iters,
                                        tol=tol,
                                        polish=True,
                                        # x0=x0,
                                        callback=callbackF,
                                        mutation=i,
                                        constraints=(nlc, nlc2, nlc3, nlc4, nlc5, nlc6, nlc7, nlc8, nlc9),
                                        workers=workers,
                                        seed=2118,  # was 235 2118 556
                                        recombination=0.7)

        end = timeit.default_timer()  # end time      #  updating='deferred'

        # %% result message and data
        if result['success'] and result['nit'] != iters:
            sys.stdout.write('\nTolerance reached at ' '%s' ' iterations.\n' % (result['nit']))
            sys.stdout.flush()
        else:
            sys.stdout.write('\n%s\n' % (result['message']))
        # %%
        y_sim = MultiGSP.run_simulations(result['x'])[0]
        # %%
        print('%s %s' % ("Objective       :", result['fun']))
        print('%s %s' % ("Design variables:", list(result['x'])))
        print("Output: ", y_sim)
        print("C1: ", constr_f(result['x']))
        print("C2: ", constr_f2(result['x']))
        print("C3: ", constr_f3(result['x']))
        print("C4: ", constr_f4(result['x']))
        print("C5: ", constr_f5(result['x']))
        print("C6: ", constr_f6(result['x']))
        print("C7: ", constr_f7(result['x']))
        print("C8: ", constr_f8(result['x']))
        print("C9: ", constr_f9(result['x']))
        print('Time            : ', end - start)  # optimisation time
        print('Function evals  : ', result['nfev'])
        print('Evals p sec     : ', result['nfev'] / (end - start))
        # %%
        plt.plot(iter_objfun)
        plt.xlabel('Iteration')
        plt.ylabel('Objective function')
        # plt.title('Genetic Algorithm')
        plt.show()
        # %%
        y_true = trueVal
        y_sim = y_sim[:len(y_true)]
        errors = 100 * (y_true - y_sim[:len(y_true)]) / (y_true + 0.000001)
        ticks = output_list[:len(y_true)]
        plt.scatter(ticks, errors, c='k', marker="*", edgecolors='r', s=80)
        plt.grid()
        plt.xlabel("Parameter")
        plt.ylabel("Error [%]")
        # plt.savefig(str(i), dpi=300)
        plt.show()
        # %%

        print("RMS             : ",
              np.sqrt(np.mean(((y_true - y_sim[:len(y_true)]) / (y_true + 0.000001)) ** 2)))

# %%

MultiGSP.terminate()

"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 2/11/2022 4:59 PM  
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
from parameters import params, params2, true_val_DP_CF6, true_val_DP_GEnx
from smithChart import rbf

# select the engine: 0 for the CF6, and 1 for the GEnx
Engine = 0
# set the GSP file names
GSPfileName = "DP2_new.mxl" if Engine == 0 else "GEnx-1B_V4DP_new.mxl"
trueVal = true_val_DP_CF6 if Engine == 0 else true_val_DP_GEnx
# all the parameters
p = params() if Engine == 0 else params2()

# %% specify inputs and outputs for GSP
# note: these should be identical to the parameters defined in the API module of GSP
if Engine == 0:
    inputs_list = ["BPR", "FPRbp", "HPCpr", 'HPC_ef', 'fan_etaC', 'HPT_ef', 'LPT_ef', 'fan_etaD', 'Cx_core', 'Cx_bp']
    output_list = ["TT25", "Ps14", "TT3", "Ps3", "Pt49", "TT49", "T5", "FN", "dH_HPC", "dH_FanC", "dH_FanBp",
                   "A_c", "A_bp", "Eta_fan", "VxHPTin", "VxHPTout", "VxLPTout", "dH_HPT", "dH_LPT"]
    #  Specify inputs for the BO function
    # bounds = [(5, 5.15), (1.5, 1.8), (10, 15),
    #           (0.85, 0.95), (0.85, 0.95), (0.85, 0.95), (0.85, 0.95), (0.85, 0.95), (0.90, 1),
    #           (0.90, 1)]  # bounds for the variables
    bounds = [(5, 5.15), (1.5, 1.8), (10, 15),
              (0.85, 0.95), (0.85, 0.95), (0.85, 0.95), (0.85, 0.95), (0.85, 0.95), (0.90, 0.95),
              (0.90, 0.95)]  # bounds for the variables

else:
    inputs_list = ["BPR", "FPRc", "FPRbp", "HPCpr", 'HPC_ef', 'fan_etaC', 'HPT_ef', 'LPT_ef', 'fan_etaD', 'Cx_core',
                   'Cx_bp']
    output_list = ["TT25", "TT3", "Ps3", "TT49", "FN", "dH_HPC", "dH_FanC", "dH_FanBp",
                   "A_c", "A_bp", "Eta_fan", "VxHPTin", "VxHPTout", "VxLPTout", "dH_HPT", "dH_LPT"]
    #  Specify inputs for the BO function
    # bounds = [(8.7, 9.1), (2.1, 2.4), (1.4, 1.7), (19, 22),
    #           (0.85, 0.95), (0.88, 0.97), (0.90, 0.97), (0.85, 0.97), (0.88, 0.97), (0.90, 1),
    #           (0.90, 1)]  # bounds for the variables

    bounds = [(8.7, 9.1), (2.1, 2.4), (1.4, 1.7), (19, 22),
              (0.85, 0.95), (0.88, 0.97), (0.85, 0.95), (0.85, 0.95), (0.88, 0.97), (0.90, 0.95),
              (0.90, 0.95)]  # bounds for the variables

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))
# import the objective function
from TestFile import objF, MultiGSP

for i in [[0, 0.6]]:  # 300 , [1, 1.5][0.0, 0.6]

    print("################### Run: " + str(i) + " #######################")

    # %% Specify inputs for the Ga function
    iters = 30  # the number of iterations or also known as generations
    pop = 15  # the population size for each generation
    tol = 0.0001  # the tolerance value for termination
    workers = 1  # amount of workers used for parallel computing
    #   x01 = [5.06860098, 1.7218356, 13.13767158, 0.8942365, 0.90479163, 0.91812425,
    #          0.86096579, 0.90484873, 0.952248, 0.94252696]
    #
    #   x011  = [5.00903616, 1.71977824, 12.79707693, 0.90175838, 0.90086607, 0.91534183,
    #         0.86581519, 0.89857935, 0.92032358, 0.94617221]  # before sm
    #
    #   x01 = [5.00990801,  1.77028897, 12.63182658,  0.90396367,  0.90260093,  0.92924971,
    # 0.89249879,  0.90220949,  0.94522948,  0.9275671]  # test cf6 75 pop 10 iter, seed = 2118
    #
    #   x000   = [5.13071296,  1.7413933,  12.818601,    0.91112676,  0.90022066, 0.89134523,
    #    0.91712047,  0.90586188,  0.92775447,  0.90892634]  # new bounds 10 pop, 30 iter, seed = 2118 (CF6)
    # For GEnx

    #   x0 = [8.73975855,  2.29210306,  1.57821232, 20.45612582,  0.89082443,  0.90784812,
    #    0.8784452,   0.9437945,   0.89779696,  0.9106631,   0.92698148] # new bounds 10 pop, 25 iter, seed = 556
    #
    #   x02 = [8.95333514,  2.35554645,  1.56775164, 19.60188997,  0.87359199,  0.90820833,
    # 0.90656291,  0.89210711,  0.92088665,  0.9462158,  0.94906007]  # for the genx (starting point)
    #
    #   x04 = [9.06301468,  2.29466272,  1.5759222,  19.86669801,  0.90116782,  0.90544465,
    # 0.92455777,  0.87494986,  0.91486836,  0.96467608,  0.95420538]

    # x0 = [8.95036027,  2.23153525,  1.57483836, 20.35359643,  0.89957973,  0.89780285,
    #  0.91478397,  0.88571894,  0.90799712,  0.9578388,   0.93028346]  # for the genx p75 i13 and no SM


    # %%  create empty lists to collect the results
    F_eval = 0
    Nfeval = 1  # iteration number
    iter_Nfeval = []  # list with iteration numbers
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
                                                                              "dH_FanC")])))  # - 0.035


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
        iter_Nfeval.append(Nfeval)
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

        # # close model and unload dll
        # if workers != 1:  # if parallel computations are done
        #     terminate()
        # cleanup(gspdll)

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
        print('Function evals  : ', F_eval)
        print('Evals p sec     : ', F_eval / (end - start))
#%%
        plt.plot(iter_objfun)
        plt.xlabel('Iteration')
        plt.ylabel('Objective function')
        # plt.title('Genetic Algorithm')
        plt.show()
#%%
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

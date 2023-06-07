import numpy as np
from GSP_helper import runGsp, loadModel, cleanup
import sys
import subprocess

sys.path.insert(1, "C:/Users/Shivan/OneDrive - Delft University of Technology/Desktop/Docs/VM/Parallel GSP/Shared "
                   "Folder/GSP")
from ParallelGSP import GSP_Server
import pickle
from Fletcher_data import loading, etaP_f, etaP_C, eta_isC
from Optimum_param import FPR
from parameters import params, true_val_DP_CF6 #, true_val_DP_GEnx, params2
from parameters_valid_testcell import true_val_DP_GEnx, params2
from threading import Thread

inputs_list, output_list, GSPfileName, Engine = pickle.load(open("io.p", "rb"))

#%% this file contains DP objective

# load local host
def path():
    subprocess.call(["C:/Users/Shivan/OneDrive - Delft University of Technology/Desktop/Docs/VM/Parallel GSP/Shared "
                     "Folder/GSP/RunLocalShiv.bat"])


Thread(target=path).start()

MultiGSP = GSP_Server(GSPfileName, N_MACHINES=1)  # Initiate GSP_Server instance
MultiGSP.connect_clients()  # Wait for all clients to connect
MultiGSP.setup_clients(inputs_list, output_list)  # Setup all clients
# gspdll = loadModel()

p = params() if Engine == 0 else params2()


def objFd(X):
    """
    The objective function
    :param X: input vector
    :return:  objective value
    """

    global output_list
    # output = runGsp(gspdll, X, output_list)
    output = MultiGSP.run_simulations(X)[0]
    obj = abs(256 - output[-1])
    # obj = 10.0 * (X[1] ** 2.0 - 6.0 * X[0] + 2.0 * X[2] ** 3.0 + X[3])
    return obj


def objF(X):
    """
    The objective function
    :param X: input vector
    :return:  objective value
    """
    global output_list
    y_sim, xp = pickle.load(open("obj.p", "rb"))
    if not set(X) == set(xp):
        X = X.tolist()
        y_sim = MultiGSP.run_simulations(X)[0]
    # y_true = np.array([0.0, 388.15, 1.507, 836.296, 32.06, 1121.65, 773.86, 256.087])
    y_true = true_val_DP_CF6 if Engine == 0 else true_val_DP_GEnx
    y_sim = np.array(y_sim[:len(y_true)])

    return np.sqrt(np.mean(((y_true - y_sim) / (y_true + 0.000001)) ** 2))

def objFBo(X):
    """
    The objective function
    :param X: input vector
    :return:  objective value
    """
    y_true = true_val_DP_CF6 if Engine == 0 else true_val_DP_GEnx

    weights = np.ones((1, len(y_true)))

    y_sim = MultiGSP.run_simulations(X)[0]
    # y_true = np.array([0.0, 388.15, 1.507, 836.296, 32.06, 1121.65, 773.86, 256.087])

    # convert the relevant polytropic efficiencies to isentropic efficiencies

    c1 = (X[inputs_list.index('HPC_ef')] - etaP_C(loading(p.inputs['N2'], p.stages, p.RmHPC,
                                                          y_sim[output_list.index("dH_HPC")] * 1000))) / X[
             inputs_list.index('HPC_ef')]
    c2 = (X[inputs_list.index("FPRbp")] - FPR(y_sim[output_list.index("Eta_fan")],
                                              X[inputs_list.index('LPT_ef')], X[inputs_list.index('BPR')],
                                              y_sim[output_list.index("FN")] * 1000, p.inputs['dot_w'], p.inputs['Ta'],
                                              p.inputs['Ma'])) / X[inputs_list.index("FPRbp")]
    c3 = (p.Abp - y_sim[output_list.index("A_bp")]) / p.Abp
    c4 = (p.Ac - y_sim[output_list.index("A_c")]) / p.Ac
    c5 = (X[inputs_list.index('fan_etaC')] - etaP_f(loading(p.inputs['N1'], 1, p.RmfanC,
                                                            y_sim[output_list.index("dH_FanC")] * 1000))) / X[
             inputs_list.index('fan_etaC')]
    c6 = (X[inputs_list.index('fan_etaD')] - etaP_f(loading(p.inputs['N1'], 1, p.RmfanBp,
                                                            y_sim[output_list.index("dH_FanBp")] * 1000))) / X[
             inputs_list.index('fan_etaD')]

    PrFcore = p.inputs['PRFcore'] if Engine == 0 else X[inputs_list.index("FPRc")]
    c7 = PrFcore * X[inputs_list.index("HPCpr")] - p.OPR  # TODO: normalise

    penc1 = 50 + abs(c1) * 100
    penc2 = 50 + abs(c2) * 100
    penc3 = 50 + abs(c3) * 100  # TODO cons are on
    penc4 = 50 + abs(c4) * 100
    penc5 = 50 + abs(c5) * 100
    penc6 = 50 + abs(c6) * 100

    y_sim = np.array(y_sim[:len(y_true)])

    return np.sqrt(np.mean((weights * (y_true - y_sim) / (y_true + 0.000001)) ** 2)) + 1 * (penc1 + penc2 + penc3 +
                                                                                            penc4 + penc5 + penc6)

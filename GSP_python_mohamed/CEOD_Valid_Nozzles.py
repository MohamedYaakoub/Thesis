import numpy as np
import pickle
from matplotlib import pyplot as plt


def scaling_F(ReDP, ReOD, a, b, c):
    """
    Scaling function is a second degree polynomial
    :param ReDP: design spool speed
    :param ReOD: off-design spool speed
    :return: function value
    """
    return np.array(c + a * ((ReOD - ReDP) / ReDP) + b * ((ReOD - ReDP) / ReDP) ** 2).reshape(-1, 1)


def nozzle_coefficients(X, DP_array, OD_array):
    Re9_DP, Re19_DP = DP_array
    Re9, Re19 = OD_array

    Cx_ub = 1.05
    Cx_lb = 0.80

    Cv_ub = 1.00
    Cv_lb = 0.95

    CX_c = scaling_F(Re9_DP, Re9, X[0], X[1], 0.9376)
    CX_c = np.clip(CX_c, Cx_lb, Cx_ub)

    CV_c = scaling_F(Re9_DP, Re9, X[2], X[3], 1)
    CV_c = np.clip(CV_c, Cv_lb, Cv_ub)

    CX_d = scaling_F(Re19_DP, Re19, X[4], X[5], 0.93)
    CX_d = np.clip(CX_d, Cx_lb, Cx_ub)

    CV_d = scaling_F(Re19_DP, Re19, X[6], X[7], 1)
    CV_d = np.clip(CV_d, Cv_lb, Cv_ub)

    # print(f'Core (Cx, Cv) {CX_c, CV_c}', f'Duct (Cx, Cv) {CX_d, CV_d}')

    return CX_c, CV_c, CX_d, CV_d

GEnx_OD, GEnx_OD_true, Alt_time, All_Reynolds = pickle.load(open(
    "Sampled flights/Reynolds_CEOD_data_mohamed_2019_feb_1-9_1_sampled.p", "rb"))

Re25_DP = All_Reynolds[0, 1]
Re3_DP = All_Reynolds[0, 2]
Re49_DP = All_Reynolds[0, 4]
Re5_DP = All_Reynolds[0, 5]
Re19_DP = All_Reynolds[0, 7]
Re9_DP = All_Reynolds[0,9]

Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19, Re6, Re9 = All_Reynolds.T


X = [-0.37501465588469474, -0.09742455801593308, -0.3390592132600383,
     -0.00781786386755412, -0.5287308619225151, 0.5753887461452354,
     -0.5562039971166222, -0.9995945647442371]

sort_Re9 = np.argsort(Re9)

plt.title("Bypass")
CX_c, CV_c, CX_d, CV_d = nozzle_coefficients(X, [Re9_DP, Re19_DP], [Re9, Re19])
plt.plot(Re9[sort_Re9], CX_c[sort_Re9], label="CX_core", marker='.')
plt.plot(Re9[sort_Re9], CV_c[sort_Re9], label="CV_core", marker='.')
plt.grid()
plt.legend()
plt.show()

sort_Re19 = np.argsort(Re19)
plt.title("Bypass")
plt.plot(Re19[sort_Re19], CX_d[sort_Re19], label="CX_bypass", marker='.')
plt.plot(Re19[sort_Re19], CV_d[sort_Re19], label="CV_bypass", marker='.')
plt.grid()
plt.legend()
plt.show()

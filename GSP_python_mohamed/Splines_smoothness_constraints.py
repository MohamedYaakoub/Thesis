import numpy as np
import pickle

file_name = "Reynolds_CEOD_data_mohamed_2019_feb_1-9_2_sampled"
_, _, _, All_Reynolds = pickle.load(open(f"Sampled flights/{file_name}.p", "rb"))

All_Reynolds_cruise = All_Reynolds[10:, :]
All_Reynolds_climb = All_Reynolds[5:10, :]
All_Reynolds_Takeoff = All_Reynolds[:5, :]

# care! DP is based on take off conditions
Re2_DP = All_Reynolds[0, 0]
Re25_DP = All_Reynolds[0, 1]
Re3_DP = All_Reynolds[0, 2]
Re4_DP = All_Reynolds[0, 3]
Re49_DP = All_Reynolds[0, 4]
Re5_DP = All_Reynolds[0, 5]
Re19_DP = All_Reynolds[0, 7]

def scaling_F(ReDP, ReOD, a, b):
    """
    Scaling function is a second degree polynomial
    :param ReDP: design spool speed
    :param ReOD: off-design spool speed
    :return: function value
    """
    return np.array(1 + a * ((ReOD - ReDP) / ReDP) + b * ((ReOD - ReDP) / ReDP) ** 2)

def scale_maps_reynolds(typef, ReDP, ReOD, file_name, poly_param):
    X = poly_param

    if typef == 'C':
        fm = scaling_F(ReDP, ReOD, X[0], X[1])
        # fpr = scaling_F(ReDP, ReOD, X[0], X[1])
        fe = scaling_F(ReDP, ReOD, X[2], X[3])
        return fm, fe
    else:
        fe = scaling_F(ReDP, ReOD, X[0], X[1])
        return fe


def get_scaling_factors_cruise_grad(X_cruise):
    Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19, _, _ = All_Reynolds_cruise.T

    Fan_c_fm, Fan_c_fe = scale_maps_reynolds("C", Re2_DP, Re2, "1_LPC_core", X_cruise[:4])
    Fan_d_fm, Fan_d_fe = scale_maps_reynolds("C", Re2_DP, Re2, "2_LPC_bypass", X_cruise[4:8])
    HPC_fm, HPC_fe = scale_maps_reynolds("C", Re25_DP, Re25, "3_HPC", X_cruise[8:12])
    HPT_fe = scale_maps_reynolds("T", Re4_DP, Re4, "4_HPT", X_cruise[12:14])
    LPT_fe = scale_maps_reynolds("T", Re49_DP, Re49, "5_LPT", X_cruise[14:16])

    Fan_c_fm_grad = numerical_differentiation(Re2, Fan_c_fm)
    Fan_c_fe_grad = numerical_differentiation(Re2, Fan_c_fe)
    Fan_d_fm_grad = numerical_differentiation(Re2, Fan_d_fm)
    Fan_d_fe_grad = numerical_differentiation(Re2, Fan_d_fe)
    HPC_fm_grad = numerical_differentiation(Re25, HPC_fm)
    HPC_fe_grad = numerical_differentiation(Re25, HPC_fe)
    HPT_fe_grad = numerical_differentiation(Re4, HPT_fe)
    LPT_fe_grad = numerical_differentiation(Re49, LPT_fe)

    return Fan_c_fm_grad, Fan_c_fe_grad, Fan_d_fm_grad, Fan_d_fe_grad, \
        HPC_fm_grad, HPC_fe_grad, HPT_fe_grad, LPT_fe_grad


def get_scaling_factors_climb_grad(X_climb):
    Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19, _, _ = All_Reynolds_climb.T

    Fan_c_fm, Fan_c_fe = scale_maps_reynolds("C", Re2_DP, Re2, "1_LPC_core", X_climb[:4])
    Fan_d_fm, Fan_d_fe = scale_maps_reynolds("C", Re2_DP, Re2, "2_LPC_bypass", X_climb[4:8])
    HPC_fm, HPC_fe = scale_maps_reynolds("C", Re25_DP, Re25, "3_HPC", X_climb[8:12])
    HPT_fe = scale_maps_reynolds("T", Re4_DP, Re4, "4_HPT", X_climb[12:14])
    LPT_fe = scale_maps_reynolds("T", Re49_DP, Re49, "5_LPT", X_climb[14:16])

    Fan_c_fm_grad = numerical_differentiation(Re2, Fan_c_fm)
    Fan_c_fe_grad = numerical_differentiation(Re2, Fan_c_fe)
    Fan_d_fm_grad = numerical_differentiation(Re2, Fan_d_fm)
    Fan_d_fe_grad = numerical_differentiation(Re2, Fan_d_fe)
    HPC_fm_grad = numerical_differentiation(Re25, HPC_fm)
    HPC_fe_grad = numerical_differentiation(Re25, HPC_fe)
    HPT_fe_grad = numerical_differentiation(Re4, HPT_fe)
    LPT_fe_grad = numerical_differentiation(Re49, LPT_fe)

    return Fan_c_fm_grad, Fan_c_fe_grad, Fan_d_fm_grad, Fan_d_fe_grad, \
        HPC_fm_grad, HPC_fe_grad, HPT_fe_grad, LPT_fe_grad


def numerical_differentiation(x, y):
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")

    n = len(x)
    dy_dx = np.zeros(n)

    # Calculate the derivative for interior points
    for i in range(1, n - 1):
        dy_dx[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])

    # Use forward/backward differences for the boundary points
    dy_dx[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy_dx[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])

    return dy_dx


def equality_constraint_cruise_climb_smoothness(X):
    X_cruise = [-0.02403791, 0.09836872, 0.09769321, -0.24873706, 0.09174088, -0.22726908,
                -0.00099752, -0.06600825, 0.06517607, -0.19915139, 0.0774644, 0.09835045,
                -0.09976, 0.09889222, 0.0987498, -0.19736836]  # used for the last 5 points
    Fan_c_fm_cruise_grad, Fan_c_fe_cruise_grad, \
        Fan_d_fm_cruise_grad, Fan_d_fe_cruise_grad, \
        HPC_fm_cruise_grad, HPC_fe_cruise_grad, \
        HPT_fe_cruise_grad, \
        LPT_fe_cruise_grad = \
        get_scaling_factors_cruise_grad(X_cruise=X_cruise)

    Fan_c_fm_climb_grad, Fan_c_fe_climb_grad, \
        Fan_d_fm_climb_grad, Fan_d_fe_climb_grad, \
        HPC_fm_climb_grad, HPC_fe_climb_grad, \
        HPT_fe_climb_grad, \
        LPT_fe_climb_grad = \
        get_scaling_factors_climb_grad(X_climb=X)

    return Fan_c_fm_climb_grad[-1] - Fan_c_fm_cruise_grad[0], Fan_c_fe_climb_grad[-1] - Fan_c_fe_cruise_grad[0], \
        Fan_d_fm_climb_grad[-1] - Fan_d_fm_cruise_grad[0], Fan_d_fe_climb_grad[-1] - Fan_d_fe_cruise_grad[0], \
        HPC_fm_climb_grad[-1] - HPC_fm_cruise_grad[0], HPC_fe_climb_grad[-1] - HPC_fe_cruise_grad[0], \
        HPT_fe_climb_grad[-1] - HPT_fe_cruise_grad[0], \
        LPT_fe_climb_grad[-1] - LPT_fe_cruise_grad[0]

def equality_Fan_c_fm_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[0]
    return constraint

def equality_Fan_c_fe_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[1]
    return constraint

def equality_Fan_d_fm_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[2]
    return constraint

def equality_Fan_d_fe_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[3]
    return constraint

def equality_HPC_fm_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[4]
    return constraint

def equality_HPC_fe_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[5]
    return constraint

def equality_HPT_fe_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[6]
    return constraint

def equality_LPT_fe_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[7]
    return constraint

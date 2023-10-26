import numpy as np
import pickle

file_name = "Reynolds_CEOD_data_mohamed_2019_feb_1-9_2_sampled"
_, _, _, All_Reynolds = pickle.load(open(f"Sampled flights/{file_name}.p", "rb"))

All_Reynolds = All_Reynolds[np.argsort(All_Reynolds[:, 0])[::-1]]

SF_design_point_TO = [0.99802762, 0.95000085, 0.99649734, 0.99076545, 0.99791594, 1.02033115,
                      0.97803967, 0.97923359]

# X_cruise = [-0.07323334, -0.01577758, 0.0976886, 0.19579827, -0.09644765, -0.03453858,
#             -0.03408711, 0.0225116, 0.02550129, -0.14322848, 0.01117573, -0.02702469,
#             -0.02245515, 0.04589156, -0.09959648, 0.09530838]

X_cruise = [-0.02223825, 0.09164879, -0.09991125, 0.19766541, -0.00923643, 0.0319769,
            - 0.09198503, 0.0346309, 0.04168849, -0.19995974, 0.01842454, -0.09209377,
            - 0.04210804, 0.0318016, -0.09989942, -0.10127006]

# care! DP is based on take off conditions
Re2_DP = All_Reynolds[0, 0]
Re25_DP = All_Reynolds[0, 1]
Re3_DP = All_Reynolds[0, 2]
Re4_DP = All_Reynolds[0, 3]
Re49_DP = All_Reynolds[0, 4]
Re5_DP = All_Reynolds[0, 5]
Re19_DP = All_Reynolds[0, 7]

Re2 = All_Reynolds[10, 0]
Re25 = All_Reynolds[10, 1]
Re3 = All_Reynolds[10, 2]
Re4 = All_Reynolds[10, 3]
Re49 = All_Reynolds[10, 4]
Re5 = All_Reynolds[10, 5]
Re19 = All_Reynolds[10, 7]


def scaling_F(ReDP, ReOD, a, b, initial_value=1):
    """
    Scaling function is a second degree polynomial
    :param ReDP: design spool speed
    :param ReOD: off-design spool speed
    :return: function value
    """
    return np.array(initial_value + a * ((ReOD - ReDP) / ReDP) + b * ((ReOD - ReDP) / ReDP) ** 2)


def scaling_F_derivative(ReDP, ReOD, a, b):
    """
    Scaling function is a second degree polynomial
    :param ReDP: design spool speed
    :param ReOD: off-design spool speed
    :return: function value
    """
    return np.array(a / ReDP + 2 * b * (ReOD - ReDP) / (ReDP ** 2))


def get_scaling(typef, ReDP, ReOD, file_name, poly_param, initial_values):
    X = poly_param

    if typef == 'C':
        fm_initial_value = initial_values[0]
        fe_initial_value = initial_values[1]

        fm = scaling_F(ReDP, ReOD, X[0], X[1], fm_initial_value)
        fm_der = scaling_F_derivative(ReDP, ReOD, X[0], X[1], )

        fe = scaling_F(ReDP, ReOD, X[2], X[3], fe_initial_value)
        fe_der = scaling_F_derivative(ReDP, ReOD, X[2], X[3])

        return fm, fm_der, fe, fe_der
    else:
        fe_initial_value = initial_values[0]

        fe = scaling_F(ReDP, ReOD, X[0], X[1], fe_initial_value)
        fe_der = scaling_F_derivative(ReDP, ReOD, X[0], X[1])
        return fe, fe_der


def get_scaling_factors_cruise(X_cruise):
    Fan_c_fm, Fan_c_fm_der, Fan_c_fe, Fan_c_fe_der = \
        get_scaling("C", Re2_DP, Re2, "1_LPC_core", X_cruise[:4], initial_values=SF_design_point_TO[0:2])

    Fan_d_fm, Fan_d_fm_der, Fan_d_fe, Fan_d_fe_der = \
        get_scaling("C", Re2_DP, Re2, "2_LPC_bypass", X_cruise[4:8], initial_values=SF_design_point_TO[2:4])

    HPC_fm, HPC_fm_der, HPC_fe, HPC_fe_der = \
        get_scaling("C", Re25_DP, Re25, "3_HPC", X_cruise[8:12], initial_values=SF_design_point_TO[4:6])

    HPT_fe, HPT_fe_der = \
        get_scaling("T", Re4_DP, Re4, "4_HPT", X_cruise[12:14], initial_values=SF_design_point_TO[6:7])

    LPT_fe, LPT_fe_der = \
        get_scaling("T", Re49_DP, Re49, "5_LPT", X_cruise[14:16], initial_values=SF_design_point_TO[7:8])

    return Fan_c_fm, Fan_c_fm_der, Fan_c_fe, Fan_c_fe_der, \
        Fan_d_fm, Fan_d_fm_der, Fan_d_fe, Fan_d_fe_der, \
        HPC_fm, HPC_fm_der, HPC_fe, HPC_fe_der, \
        HPT_fe, HPT_fe_der, \
        LPT_fe, LPT_fe_der


def get_scaling_factors_climb(X_climb):
    Fan_c_fm, Fan_c_fm_der, Fan_c_fe, Fan_c_fe_der = \
        get_scaling("C", Re2_DP, Re2, "1_LPC_core", X_climb[:4], initial_values=SF_design_point_TO[0:2])

    Fan_d_fm, Fan_d_fm_der, Fan_d_fe, Fan_d_fe_der = \
        get_scaling("C", Re2_DP, Re2, "2_LPC_bypass", X_climb[4:8], initial_values=SF_design_point_TO[2:4])

    HPC_fm, HPC_fm_der, HPC_fe, HPC_fe_der = \
        get_scaling("C", Re25_DP, Re25, "3_HPC", X_climb[8:12], initial_values=SF_design_point_TO[4:6])

    HPT_fe, HPT_fe_der = \
        get_scaling("T", Re4_DP, Re4, "4_HPT", X_climb[12:14], initial_values=SF_design_point_TO[6:7])

    LPT_fe, LPT_fe_der = \
        get_scaling("T", Re49_DP, Re49, "5_LPT", X_climb[14:16], initial_values=SF_design_point_TO[7:8])

    return Fan_c_fm, Fan_c_fm_der, Fan_c_fe, Fan_c_fe_der, \
        Fan_d_fm, Fan_d_fm_der, Fan_d_fe, Fan_d_fe_der, \
        HPC_fm, HPC_fm_der, HPC_fe, HPC_fe_der, \
        HPT_fe, HPT_fe_der, \
        LPT_fe, LPT_fe_der


def equality_constraint_cruise_climb_smoothness(X):
    Fan_c_fm_cruise, Fan_c_fm_der_cruise, Fan_c_fe_cruise, Fan_c_fe_der_cruise, \
        Fan_d_fm_cruise, Fan_d_fm_der_cruise, Fan_d_fe_cruise, Fan_d_fe_der_cruise, \
        HPC_fm_cruise, HPC_fm_der_cruise, HPC_fe_cruise, HPC_fe_der_cruise, \
        HPT_fe_cruise, HPT_fe_der_cruise, \
        LPT_fe_cruise, LPT_fe_der_cruise = \
        get_scaling_factors_cruise(X_cruise=X_cruise)

    Fan_c_fm_climb, Fan_c_fm_der_climb, Fan_c_fe_climb, Fan_c_fe_der_climb, \
        Fan_d_fm_climb, Fan_d_fm_der_climb, Fan_d_fe_climb, Fan_d_fe_der_climb, \
        HPC_fm_climb, HPC_fm_der_climb, HPC_fe_climb, HPC_fe_der_climb, \
        HPT_fe_climb, HPT_fe_der_climb, \
        LPT_fe_climb, LPT_fe_der_climb = \
        get_scaling_factors_climb(X_climb=X)

    # print(HPT_fe_climb - HPT_fe_cruise)

    return Fan_c_fm_climb - Fan_c_fm_cruise, Fan_c_fe_climb - Fan_c_fe_cruise, \
           Fan_d_fm_climb - Fan_d_fm_cruise, Fan_d_fe_climb - Fan_d_fe_cruise, \
           HPC_fm_climb - HPC_fm_cruise, HPC_fe_climb - HPC_fe_cruise, \
           HPT_fe_climb - HPT_fe_cruise, \
           LPT_fe_climb - LPT_fe_cruise, \
           Fan_c_fm_der_climb - Fan_c_fm_der_cruise, Fan_c_fe_der_climb - Fan_c_fe_der_cruise, \
           Fan_d_fm_der_climb - Fan_d_fm_der_cruise, Fan_d_fe_der_climb - Fan_d_fe_der_cruise, \
           HPC_fm_der_climb - HPC_fm_der_cruise, HPC_fe_der_climb - HPC_fe_der_cruise, \
           HPT_fe_der_climb - HPT_fe_der_cruise, \
           LPT_fe_der_climb - LPT_fe_der_cruise,


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


# derivatives start here

def equality_Fan_c_fm_der_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[8]
    return constraint


def equality_Fan_c_fe_der_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[9]
    return constraint


def equality_Fan_d_fm_der_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[10]
    return constraint


def equality_Fan_d_fe_der_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[11]
    return constraint


def equality_HPC_fm_der_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[12]
    return constraint


def equality_HPC_fe_der_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[13]
    return constraint


def equality_HPT_fe_der_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[14]
    return constraint


def equality_LPT_fe_der_cruise_climb(X):
    constraint = equality_constraint_cruise_climb_smoothness(X=X)[15]
    return constraint

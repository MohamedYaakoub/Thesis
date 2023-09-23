import pickle
import numpy as np
from GSP_helper import cleanup, runGsp
from matplotlib import pyplot as plt

Engine = 1  # Enter zero for the CF6 and 1 for the GEnx
GSPfileName = "OffDesignGEnx_Valid_All parameters.mxl"  # "GEnx-1B_V3_test2.mxl"  #

file_name = "Reynolds_CEOD_data_mohamed_2019_feb_1-9_1_sampled"
GEnx_OD, GEnx_OD_true, _, All_Reynolds = pickle.load(open(f"Sampled flights/{file_name}.p", "rb"))

OG_list = ["N1", "P0", "T0", "Mach", "HP"]
OD_input_list = ["ODSF_Wc core", "ODSF_PR core", "ODSF_Eta core",  # Fan core
                 "ODSF_Wc duct", "ODSF_PR duct", "ODSF_Eta duct",  # Fan duct
                 "ODSF_Wc", "ODSF_PR", "ODSF_Eta",  # HPC
                 "ODSF_Wc", "ODSF_PR", "ODSF_Eta",  # HPT
                 "ODSF_Wc", "ODSF_PR", "ODSF_Eta"]  # LPT

inputs_list = OG_list + OD_input_list
output_list = ["TT25", "TT3", "Ps3", "TT49", "Wf", "N2", "Re2", "Re25", "Re3", "Re4", "Re49", "Re5", "Re14", "Re19"]

pickle.dump([inputs_list, output_list, GSPfileName, Engine], open("io.p", "wb"))

# GEnx_OD = np.array([item for sublist in GEnx_OD for item in sublist])
# GEnx_OD_true = np.array([item for sublist in GEnx_OD_true for item in sublist])


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
        fm = scaling_F(ReDP, ReOD, X[2], X[3])
        fpr = scaling_F(ReDP, ReOD, X[0], X[1])
        fe = scaling_F(ReDP, ReOD, X[4], X[5])
        return fm, fpr, fe
    else:
        fe = scaling_F(ReDP, ReOD, X[0], X[1])
        return fe


Fan_c_fm, Fan_c_fpr, Fan_c_fe = 1, 1, 1
Fan_d_fm, Fan_d_fpr, Fan_d_fe = 1, 1, 1
HPC_fm, HPC_fpr, HPC_fe = 1, 1, 1
HPT_fe = 1
LPT_fe = 1

OD_scaling_array = np.array([1, 1, 1,
                             1, 1, 1,
                             1, 1, 1,
                             1, 1, 1,
                             1, 1, 1])

OD_scaling_array2 = np.array([1, 1.5, 1.5,
                             1, 1, 2,
                             1, 2, 2,
                             1, 2, 2,
                             1, 1, 2])



run_gsp_input = np.concatenate((GEnx_OD, np.tile(OD_scaling_array, (len(GEnx_OD),1))), axis=1)
run_gsp_input2 = np.concatenate((GEnx_OD, np.tile(OD_scaling_array2, (len(GEnx_OD),1))), axis=1)

# run_gsp_input2 = np.ones((15,20))

from my_modified_functions import gspdll
def run_simulation(input):
    gspdll.InitializeModel()
    y_sim = np.array(runGsp(gspdll, input, output_list))
    return y_sim[:, :6]

run1 = run_simulation(run_gsp_input)
run2 = run_simulation(run_gsp_input2)

for i in range(len(run1[0])):  # 0: takeoff 1:climb 2:cruise
    plt.scatter(GEnx_OD[:, 0], run1[:, i], label=output_list[i + 6] + " unmodified")
    plt.scatter(GEnx_OD[:, 0], run2[:, i], label=output_list[i + 6] + " modified")
    plt.xlabel('Corrected Fan Speed [%]')
    plt.ylabel(str(i))
    plt.legend()
    plt.show()

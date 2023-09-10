import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
import sys

# GEnx_OD, GEnx_OD_true, All_Reynolds = pickle.load(open("../Clusters/Reynolds_one_flight_sampled.p", "rb"))
# Re2, Re25, Re3, Re4, Re49, Re5, Re14, Re19 = All_Reynolds.T

# print(Re25.shape)



# GEnx_OD, GEnx_OD_true, N1cCEOD = pickle.load(open("../CEOD_GEnx/CEOD_set_Valid.p", "rb"))
#
# _, All_Reynolds = pickle.load(open("../Constants/Reynolds_set_Valid.p", "rb"))

GEnx_OD, GEnx_OD_true, All_Reynolds = pickle.load(open("../Clusters/Reynolds_input_clusters.p", "rb"))
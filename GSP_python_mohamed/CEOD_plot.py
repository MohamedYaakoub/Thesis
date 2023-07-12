import pickle
import numpy as np
import timeit
import sys
from GSP_helper import cleanup, runGsp

from matplotlib import pyplot as plt

from map_functions import reset_maps

file_name = "CEOD_160724-193429-KLM891____-EHAMZUUU-KL_PH-BHA-2-956609-W007FFD.p"
file_name_set_valid = "CEOD_set_valid.p"

GEnx_OD, GEnx_OD_true, _ = pickle.load(open("CEOD_GEnx/" + file_name, "rb"))
GEnx_OD_set_valid, GEnx_OD_true_set_valid, _ = pickle.load(open("CEOD_GEnx/" + file_name_set_valid, "rb"))

idx = 2
plt.scatter(GEnx_OD[0][:, 0], GEnx_OD[0][:, idx], alpha=1)
plt.scatter(GEnx_OD_set_valid[0][:, 0], GEnx_OD_set_valid[0][:, idx], alpha=0.3, color="black")
plt.title("Take off")
plt.show()

plt.scatter(GEnx_OD[1][:, 0], GEnx_OD[1][:, idx], alpha=1)
plt.scatter(GEnx_OD_set_valid[1][:, 0], GEnx_OD_set_valid[1][:, idx], alpha=0.3, color="black")
plt.title("Climb")
plt.show()

print(f'ranges {np.min(GEnx_OD[1][:, idx]), np.max(GEnx_OD[1][:, idx])}')
print(f'set valid ranges {np.min(GEnx_OD_set_valid[1][:, idx]), np.max(GEnx_OD_set_valid[1][:, idx])}')
print(f'Take off {GEnx_OD[0].shape}')
print(f'Climb {GEnx_OD[1].shape}')

plt.scatter(GEnx_OD[2][:, 0], GEnx_OD[2][:, idx], alpha=1)
plt.scatter(GEnx_OD_set_valid[2][:, 0], GEnx_OD_set_valid[2][:, idx], alpha=0.3, color="black")
plt.title("cruise")
plt.show()
print(f'Cruise {GEnx_OD[2].shape}')

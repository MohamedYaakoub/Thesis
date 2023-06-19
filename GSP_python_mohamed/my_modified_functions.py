import pickle
import ctypes
import os
import sys

import numpy as np
from _ctypes import FreeLibrary


curr_doler_path = os.getcwd()
def loadModel(Engine, name):
    # print("Loading API and model . . .", end='\r')
    print("Loading API and model . . .")
    gspdll = ctypes.cdll.LoadLibrary(curr_doler_path +"\GSP.dll")  # load DLL for API
    if Engine == 0:
        # gspdll.LoadModel("CF6-80C2/" + name, 0)  # load the GSP model
        sys.exit()
    else:
        print("Loading C:/Users/mohsy/University/KLM/Thesis/My thesis/Code/GSP_python_mohamed/GSP_files/" + name)
        gspdll.LoadModel("C:/Users/mohsy/University/KLM/Thesis/My thesis/Code/GSP_python_mohamed/GSP_files/" + name, 0)  # load the GSP model
    gspdll.ConfigureModel()  # Sets up model for API operation

    print("\rLoading Complete")
    return gspdll


inputs_list, output_list, GSPfileName, Engine = pickle.load(open("io.p", "rb"))

gspdll = loadModel(Engine, GSPfileName)
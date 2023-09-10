"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 2/13/2022 11:25 AM  
"""

import ctypes as ctypes
import os

import numpy as np
from _ctypes import FreeLibrary

# %%
def runGsp(gspdll, inputs, outputs):
    # if isinstance(inputs[0], float):
    #     inputs = [inputs]
    # settings model, these input parameters should be specified in the GSP API

    # if inputs.ndim > 1:
    if not isinstance(inputs[0], float):
        Results = [0] * len(inputs)
        for ID, inp in enumerate(inputs):
            for i in range(0, len(inp)):
                gspdll.SetInputControlParameterByIndex(i + 1, ctypes.c_double(inp[i]))
            # the RunModel function used below has four inputs:
            # - Boolean input parameter to show (true) or hide (false) the start time dialog
            # - Boolean input parameter to show (true) or hide (false) the stabilise dialog
            # - Boolean input parameter to stabilise the simulation (true) at the current time. I.e. a
            # steady state calculation for the current input conditions will be calculated.
            # - Boolean input parameter to show (true) or hide (false) the progress bar window.
            # gspdll.RunModel(0, 0, 0, 0)  # run the gsp model
            gspdll.CalculateSteadyStatePoint(0, 0)
            # this is the output from the model, as specified in GSP API (the same order)
            output_set = []  # collect all the specified outputs in this list
            for j in range(1, len(outputs)+1):
                dv = ctypes.c_double()
                # gspdll.GetOutputDataParameterValueByIndex(j, ctypes.pointer(dv), 0)
                string_dummy = ctypes.create_string_buffer(b"", 256)
                gspdll.GetOutputDataParameterByIndex(j, ctypes.byref(string_dummy), ctypes.byref(dv), 0)
                output_set.append(dv.value)
            Results[ID] = output_set
    else:
        inp = inputs
        for i in range(0, len(inp)):
            gspdll.SetInputControlParameterByIndex(i + 1, ctypes.c_double(inp[i]))
        # the RunModel function used below has four inputs:
        # - Boolean input parameter to show (true) or hide (false) the start time dialog
        # - Boolean input parameter to show (true) or hide (false) the stabilise dialog
        # - Boolean input parameter to stabilise the simulation (true) at the current time. I.e. a
        # steady state calculation for the current input conditions will be calculated.
        # - Boolean input parameter to show (true) or hide (false) the progress bar window.
        # gspdll.RunModel(0, 0, 0, 0)  # run the gsp model
        gspdll.CalculateSteadyStatePoint(0, 0)
        # this is the output from the model, as specified in GSP API (the same order)
        output_set = []  # collect all the specified outputs in this list
        for j in range(1, len(outputs) + 1):
            dv = ctypes.c_double()
            # gspdll.GetOutputDataParameterValueByIndex(j, ctypes.pointer(dv), 0)

            string_dummy = ctypes.create_string_buffer(b"", 256)
            gspdll.GetOutputDataParameterByIndex(j, ctypes.byref(string_dummy), ctypes.byref(dv), 0)
            output_set.append(dv.value)
        Results = output_set

    return np.array(Results)

# %%

curr_doler_path = os.getcwd()

def loadModel(Engine, name):
    print("Loading API and model . . .", end='\r')
    gspdll = ctypes.cdll.LoadLibrary(curr_doler_path +"\GSP.dll")  # load DLL for API
    if Engine == 0:
        gspdll.LoadModel("CF6-80C2/" + name, 0)  # load the GSP model
    else:
        gspdll.LoadModel("GENX-1B_model/" + name, 0)  # load the GSP model
    gspdll.ConfigureModel()  # Sets up model for API operation

    print("\rLoading Complete")
    return gspdll

# %% Cleanup, called when program finishes/closes
def cleanup(gspdll):
    pass
    # This extra work for unloading the dll is done to allow GSP to actually open
    # If this unloading was not done, GSP thinks it is still supposed to connect using the API
    # Opening GSP would then be invisible until performing a full kernel restart of Python
    gspdll.CloseModel(True) # Close model, don't show save dialog
    gspdll.FreeAll()  # Free dll before unloading
    FreeLibrary(gspdll._handle)  # Unload dll



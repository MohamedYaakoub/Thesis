# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:22:23 2021
A self-implemented GSP API connection able to run a single or multiple simulations
Goal is an easy to understand, easy to use wrapper to investigate relations
@author: BramB
"""

import ctypes
from _ctypes import FreeLibrary
from matplotlib import pyplot as plt
import numpy as np
import os
import time

# %% Settings
# User defined settings, free to change
Settings = {
    'N%1': 96.96,                           # Shaft 1 speed as percentage
    'RH' : 71.22,                           # Relative humidity
    'Ts' : 285.59,                          # Ambient static temperature
    'Ps' : 0.9867,                          # Ambient static pressure
    'LPC_duct_eff': 0, 'LPC_duct_Wc': 0,    # LPC duct (fan+bypass)
    'LPC_core_eff': 0, 'LPC_core_Wc': 0,    # LPC core (fan+LPC)
    'HPC_eff': 0, 'HPC_Wc': 0,              # HPC
    'HPT_eff': 0, 'HPT_Wc': 0,              # HPT
    'LPT_eff': 0, 'LPT_Wc': 0               # LPT
    }

# Ordered inputs as expected by GSP, only change when changing API block in GSP
InputList = [
    'N%1',
    'RH',
    'Ts',
    'Ps',
    'LPC_duct_eff', 'LPC_duct_Wc',
    'LPC_core_eff', 'LPC_core_Wc',
    'HPC_eff', 'HPC_Wc',
    'HPT_eff', 'HPT_Wc',
    'LPT_eff', 'LPT_Wc'
    ]

# Ordered output list as output by GSP, only change when changing API block in GSP
OutputList = [
    'N%2',
    'WF',
    'TT25',
    'TT3',
    'PS3',
    'TT49'
    ]

# Ordered run list, overwrites the value in 'Settings' for all runs 
# If not defined (zero length or only name entry), a single simulation is ran with the values in 'Settings'
# First element of list is the value considered (string), subsequent entries are simulated settings (floats)
RunList = ['N%1'] + np.linspace(50,110,50).tolist()
    

# %% Main function, called if file is ran directly
def main(Settings,InputList,OutputList,RunList):
    print("Loading API and model")
    MainPath = os.getcwd() # Path to this python file
    gspdll = ctypes.cdll.LoadLibrary(MainPath + "\\API\\GSP.dll") # Load dll
    gspdll.LoadModel(MainPath + "\\Model-0\\GEnx-1B_V3_Single.mxl", False) # Load model
    gspdll.ConfigureModel()  # Sets up model for API operation
   
    t_start = time.time()
    print("Initializing and running model")
    # Check if IterList containst a string and has at least a single numerical value
    Output = []
    if len(RunList) > 2 and isinstance(RunList[0],str):
        print("Conducting multiple runs for setting {}".format(RunList[0]))
        run_counter = 0
        for i in RunList[1:]: # Loop over RunList, skip first element since it is the dictionary key
            run_counter += 1
            print("\rSimulating: {:03d}/{:03d}".format(run_counter,len(RunList)-1),end='\r')
            Settings[RunList[0]] = i # Overwrite standard value in settings dict with value in RunList
            set_inputs(gspdll,InputList,Settings) # Set inputs for run
            Output.append(get_results(gspdll, OutputList)) # Run model and append results to Output list
        print('\rSimulating: DONE   ')
        plot_results(OutputList, RunList, Output)
        
    else:
        print("Conducting single run\n")
        set_inputs(gspdll,InputList,Settings) # Set inputs for run
        Output = get_results(gspdll, OutputList) # Run model and append results to Output list
        for i in range(0,len(Output)):
            print("{:<5}: {}".format(OutputList[i],round(Output[i],2)))
    print("Simulation time: {}".format(round(time.time() - t_start, 2)))
    cleanup(gspdll)
    return Output


# %% Sets the inputs
def set_inputs(gspdll,InputList,Settings):
    # Check if expected amount of inputs and list of inputs is the same
    n_input = ctypes.c_int();
    gspdll.GetInputControlParameterArraySize(ctypes.byref(n_input))  
    if n_input.value != len(InputList):
        raise AssertionError("Amount of expected inputs by GSP and input list are not of equal size.")
    
    for i in range(0,len(Settings)):
        gspdll.SetInputControlParameterByIndex(i+1, ctypes.c_double(Settings[InputList[i]]))    


# %% Runs model, fetches model run results, returns an array of the results
def get_results(gspdll,OutputList):
    # Check if expected amount of outputs and list of outputs is the same
    n_output = ctypes.c_int();
    gspdll.GetOutputDataListSize(ctypes.byref(n_output),None)   
    if n_output.value != len(OutputList):
        raise AssertionError("Amount of expected outputs by GSP and output list are not of equal size.")
    
    gspdll.RunModel(False,False,False,False) # Run model
    
    # Fetch all of the outputs, append to an array
    outputs = []
    for i in range(0,n_output.value):
        temp = ctypes.c_double()
        gspdll.GetOutputDataParameterValueByIndex(i+1,ctypes.byref(temp),False)
        outputs.append(temp.value)
    return outputs


# %% Plot results if multiple runs were conducted
def plot_results(OutputList,RunList,Output):
    print("Plotting results")
    x = RunList[1:]
    for i in range(0,len(OutputList)):
        y = []
        for ii in range(0,len(Output)):
            y.append(Output[ii][i])
        plt.figure()
        plt.plot(x,y,linewidth=3)
        plt.xlabel(RunList[0])
        plt.ylabel(OutputList[i])
        plt.title("Simulation results for {} for varying {}".format(OutputList[i],RunList[0]))
        plt.grid(True)
    return


# %% Cleanup, called when program finishes/closes
def cleanup(gspdll):
    # This extra work for unloading the dll is done to allow GSP to actually open
    # If this unloading was not done, GSP thinks it is still supposed to connect using the API
    # Opening GSP would then be invisible untill performing a full kernel restart of Python
    gspdll.CloseModel(True) # Close model, don't show save dialog
    gspdll.FreeAll() # Free dll before unloading
    FreeLibrary(gspdll._handle) # Unload dll


# %% If file is ran directly, call main function
if __name__ == "__main__":
    Output = main(Settings,InputList,OutputList,RunList)
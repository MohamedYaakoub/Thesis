import ctypes
import os
import sys
from _ctypes import FreeLibrary


def loadModel(name):
    curr_path = os.getcwd()
    # print("Loading API and model . . .", end='\r')
    print("Loading API and model . . .")
    gspdll = ctypes.cdll.LoadLibrary(curr_path + "\GSP.dll")  # load DLL for API

    print(f"Loading {curr_path}" + "\\" + name)
    gspdll.LoadModel(curr_path.replace("\\", '/') + '/' + name, 0)  # load the GSP model

    gspdll.ConfigureModel()  # Sets up model for API operation

    print("\rLoading Complete")
    return gspdll


def runGsp(gspdll, inputs, outputs):
    if isinstance(inputs[0], float) or isinstance(inputs[0], int):
        inputs = [inputs]
    Results = [0] * len(inputs)
    # settings model, these input parameters should be specified in the GSP API
    for ID, inp in enumerate(inputs):
        print(inp)
        for i in range(0, len(inp)):
            gspdll.SetInputControlParameterByIndex(i + 1, ctypes.c_double(inp[i]))

        # the RunModel function used below has four inputs:
        # - Boolean input parameter to show (true) or hide (false) the start time dialog
        # - Boolean input parameter to show (true) or hide (false) the stabilise dialog
        # - Boolean input parameter to stabilise the simulation (true) at the current time. I.e. a
        # steady state calculation for the current input conditions will be calculated.
        # - Boolean input parameter to show (true) or hide (false) the progress bar window.

        gspdll.RunModel(0, 0, 0, 0)  # run the gsp model
        # gspdll.CalculateSteadyStatePoint(0, 0)

        # this is the output from the model, as specified in GSP API (the same order)
        output_set = []  # collect all the specified outputs in this list
        for j in range(1, len(outputs) + 1):
            # gspdll.GetOutputDataParameterValueByIndex(j, ctypes.byref(dv), 0)
            p = ctypes.c_double()
            string_dummy = ctypes.create_string_buffer(10)
            x = gspdll.GetOutputDataParameterByIndex(j, ctypes.byref(string_dummy), ctypes.byref(p), 0)
            # x = gspdll.GetOutputDataParameterValueByIndex(j, True)
            print(x)
            output_set.append(p.value)
        Results[ID] = output_set
    return Results


def run():
    input_datapoint = [1.1, 0.38]  # ODSF_Eta, Wf
    output_list = ["FN"]

    GSPfileName = "TJET_API.mxl"
    # GSPfileName = "TJET_API_no_OD_scaling.mxl"

    gspdll = loadModel(GSPfileName)

    y_sim = runGsp(gspdll, input_datapoint, output_list)

    print(f'Simulated Force {y_sim}')

run()
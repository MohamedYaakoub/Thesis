# -*- coding: utf-8 -*-
"""
Created on Mon Jan 07 2022
Implementation of a parallel GSP method. Server connects to an arbitrary number
of clients and designates simulations to these clients.
CLIENT

@author: BramB
"""

import socket
import sys
import pickle
import os
import ctypes
from _ctypes import FreeLibrary

# %% Settings
PORT = 1919  # The port used by the system

if len(sys.argv) == 2:
    ID = int(sys.argv[1])
    print("The client will use ID {}".format(ID))
else:
    print("Please provide system ID as script argument")
    sys.exit()

if ID == 10:
    HOST = 'localhost'
else:
    HOST = '10.0.2.2'


# %% Termination function, called on close
def terminate(gspdll, client):
    # This extra work for unloading the dll is done to allow GSP to actually open
    # If this unloading was not done, GSP thinks it is still supposed to connect using the API
    # Opening GSP would then be invisible untill performing a full kernel restart of Python
    gspdll.CloseModel(True)  # Close model, don't show save dialog
    gspdll.FreeAll()  # Free dll before unloading
    FreeLibrary(gspdll._handle)  # Unload dll
    client.close()  # Close the connection
    sys.exit()


# %% Set up model and perform checks
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Looking for server...")

connected = False
while not connected:
    try:
        client.connect((HOST, PORT))
    except:
        continue
    else:
        connected = True
        client.send(str(ID).encode())
        print("Connected to server!")

# %% Receive inputs and outputs lists
print("Receiving InputList and OutputList")
InputList = pickle.loads(client.recv(4096))
client.sendall(pickle.dumps(InputList))

OutputList = pickle.loads(client.recv(4096))
client.sendall(pickle.dumps(OutputList))

print("Starting GSP")
try:
    MainPath = os.getcwd()  # Path to this python file
    gspdll = ctypes.cdll.LoadLibrary(MainPath + "\\API\\GSP.dll")  # Load dll
    gspdll.LoadModel(MainPath + "\\Model-{}\\DP2.mxl".format(ID), False)  # Load model
    gspdll.ConfigureModel()  # Sets up model for API operation
except:
    print("Error loading GSP model")
    client.sendall(b"0")  # Let the server know that GSP failed to load
    terminate(gspdll, client)
else:
    print("GSP and model loaded")
    client.sendall(b"1")  # Let the server know that GSP is loaded

# Check if the in- and output list sizes match GSP expectations
n_input = ctypes.c_int()
n_output = ctypes.c_int()
gspdll.GetInputControlParameterArraySize(ctypes.byref(n_input))
gspdll.GetOutputDataListSize(ctypes.byref(n_output), None)

# Send OK to server
if n_input.value == len(InputList) and n_output.value == len(OutputList):
    print("Model setup succesfully completed, running simulation requests")
    client.sendall(b'1')
else:
    print("In- and/or outputlist did not match GSP expectations, exiting")
    client.sendall(b'0')
    terminate(gspdll, client)

# %% Complete simulation requests
while True:
    Request = client.recv(4096)
    # If the server is closed it reads an empty string, when this happens terminate
    if Request:
        Request = pickle.loads(Request)  # Decode
        # Set all inputs in GSP
        for i in range(0, len(Request)):
            gspdll.SetInputControlParameterByIndex(i + 1, ctypes.c_double(Request[i]))

        # Run model and retrieve results
        gspdll.RunModel(False, False, False, False)
        outputs = []
        for i in range(0, n_output.value):
            temp = ctypes.c_double()
            gspdll.GetOutputDataParameterValueByIndex(i + 1, ctypes.byref(temp), False)
            outputs.append(temp.value)

        # Send results back to server
        client.sendall(pickle.dumps(outputs))

    else:
        print("Server closed, terminating")
        terminate(gspdll, client)

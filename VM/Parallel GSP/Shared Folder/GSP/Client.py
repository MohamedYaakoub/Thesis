# -*- coding: utf-8 -*-
"""
Created on Mon Jan 07 2022
Client side of a parallel GSP implementation.
Connects to server, runs required setup, receives simulation request and sends
results back. Can run arbitrary number of requests, stops execution when 
server closes.

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
BUFSIZE = 4096  # Buffer size used by sockets

if len(sys.argv) == 2:
    ID = int(sys.argv[1])
    print("The client will use ID {}".format(ID))
else:
    print("Please provide system ID as script argument")
    sys.exit()

# If the client is ran using ID 0, it is ran locally, use localhost as HOST
if ID == 0:
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

# %% Function for initialisation of the model
def initialize(gspdll):
    gspdll.InitializeModel()

# %% Set up model and perform checks
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Looking for server...")

connected = False

# Keep listening for server while not connected
# .connect() can time and error out, therefore the try and except
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

InputList = pickle.loads(client.recv(BUFSIZE))
client.sendall(pickle.dumps(InputList))

OutputList = pickle.loads(client.recv(BUFSIZE))
client.sendall(pickle.dumps(OutputList))

GSPfileName = pickle.loads(client.recv(BUFSIZE))

print("Starting GSP")
try:
    MainPath = "C:\\Users\\Shivan\\OneDrive - Delft University of Technology\\Desktop\\Docs\\VM\\Parallel GSP\\" \
    "Shared Folder\\GSP"
    # MainPath = os.getcwd()  # Path to this python file
    gspdll = ctypes.cdll.LoadLibrary(MainPath + "\\API\\GSP.dll")  # Load dll
    gspdll.LoadModel(MainPath + "\\Model-{}\\".format(ID) + GSPfileName, 0)  # Load model
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
    print("\033[0;32m" + "\033[3m" + "Model setup succesfully completed, running simulation requests" + "\033[0;0m")
    client.sendall(b'1')
else:
    print("\033[0;31m" + "\033[3m" + "In- and/or outputlist did not match GSP expectations, exiting"  + "\033[0;0m")
    client.sendall(b'0')
    terminate(gspdll, client)

# %% Complete simulation requests
while True:

    Request = client.recv(BUFSIZE)
    # If the server is closed it reads an empty string, when this happens terminate
    if Request:
        Request = pickle.loads(Request)  # Decode
        if isinstance(Request, str) and Request == 'INIT':
            gspdll.InitializeModel()
            continue
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


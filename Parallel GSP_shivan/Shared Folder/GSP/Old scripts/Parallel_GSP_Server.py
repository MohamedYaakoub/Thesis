# -*- coding: utf-8 -*-
"""
Created on Mon Jan 07 2022
Implementation of a parallel GSP method. Server connects to an arbitrary number
of clients and designates simulations to these clients. 
SERVER
Version 1.0
@author: BramB
"""

import socket
import sys
import select
import numpy as np
import pickle
from matplotlib import pyplot as plt
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

# Networking settings
HOST = ""       # Listen on all adresses
PORT = 1919     # Port to listen on
N_MACHINES = 3  # Number of clients expected

# State tracker of machine setup
state = np.zeros((N_MACHINES,5))


# %% Termination function, called on close
def terminate(server,conns):
    # Properly shutdown and close server
    for conn in conns:
        conn.close()
        
    server.close()
    sys.exit()

# %% Wait for client connection, save connections to conns with position equal to ID
def connect_clients(server, n_machines):
    global state
    conns = [0]*n_machines
    
    show_state_setup(state)
    while True:
        server.listen(n_machines)
        conn, _ = server.accept()
        
        ID = int(conn.recv(4096))
        if not conns[ID]:
            # ID is new, add to conns list
            conns[ID] = conn
            state[ID, 0] = 1
            show_state_setup(state)
        
        if all(conns):
            return conns
    
# %% Prints the state during the setup process
def show_state_setup(state):
    print("\033[H\033[J") # Clears the console
    print("Parallel-GSP Setup Sequence")
    print("___________________________\n")
    
    id_string = ""
    connected_string = ""
    inputlist_string = ""
    outputlist_string = ""
    GSP_string = ""
    setupcheck_string = ""
    
    ID = 0
    for i in state:
        id_string += " " + str(ID) + "‍ "
        
        if i[0]:
            connected_string += "🟩 "
        else:
            connected_string += "🟥 "

        if i[1]:
            inputlist_string += "🟩 "
        else:
            inputlist_string += "🟥 "           
            
        if i[2]:
            outputlist_string += "🟩 "
        else:
            outputlist_string += "🟥 "

        if i[3]:
            GSP_string += "🟩 "
        else:
            GSP_string += "🟥 "
            
        if i[4]:
            setupcheck_string += "🟩 "
        else:
            setupcheck_string += "🟥 "
                
        ID += 1

    print(" Client ID | " + id_string)
    print(" Connected | " + connected_string)
    print(" InputList | " + inputlist_string)
    print("OutputList | " + outputlist_string)
    print(" GSP ready | " + GSP_string)
    print("  Setup OK | " + setupcheck_string)

# %% Send settings to clients and start GSP
def setup_clients(conns, InputList, OutputList):
    global state
    
    # Send input list to all clients
    for ID, conn in enumerate(conns):
        conn.sendall(pickle.dumps(InputList))
        
        # Client sends list back to check if connection is solid
        if pickle.loads(conn.recv(4096)) != InputList:
            print("InputList was not properly transferred to client {}".format(ID))
            terminate(server,conn)
        else:
            state[ID, 1] = 1
            show_state_setup(state)
        
    # Send output lists to all clients
    for ID, conn in enumerate(conns):
        conn.sendall(pickle.dumps(OutputList))
        
        # Client sends list back to check if connection is solid
        if pickle.loads(conn.recv(4096)) != OutputList:
            print("OutputList was not properly transferred to client {}".format(ID))
            terminate(server,conn)
        else:
            state[ID, 2] = 1
            show_state_setup(state)
            
    # Wait for all the clients to finish loading GSP and giving OK
    while not all(state[:,4]):
        readable, _, _ = select.select(conns, [], [])
        for conn in readable:
            ID = conns.index(conn)
            if state[ID, 3] == 0:
                # GSP loading message
                if int(conn.recv(4096)):
                    state[ID, 3] = 1
                    show_state_setup(state)
                else:
                    print("Client {} was unable to load GSP, exiting".format(ID))
                    terminate(server,conn)
            else:
                # Input/Output count message
                if int(conn.recv(4096)):
                    state[ID, 4] = 1
                    show_state_setup(state)
                else:
                    print("Client {} does not have a matching GSP model, exiting".format(ID))
                    terminate(server,conn)
    
    print("\nClient setup succesfully completed")

# %% Prints the state during the simulating process
def show_state_simulating(states, last_completed):
    print("\033[H\033[J") # Clears the console
    print("Parallel-GSP Simulating")
    print("___________________________\n")
    
    id_string = ""
    states_string = ""
    last_completed_string = ""
    n_completed_string = ""
    for ID in range(0, len(states)):
        id_string += "  " + str(ID) + "‍  | "
        states_string += " " + "{:03d}".format(states[ID]) + " | "
        last_completed_string += " " + "{:03d}".format(last_completed[ID]) + " | "
        n_completed_string += " " + "{:03d}".format(n_completed[ID]) + " | "
        
    print("        Client ID | " + id_string)
    print("Currently running | " + states_string)
    print("   Last completed | " + last_completed_string)
    print(" Amount completed | " + n_completed_string)

# %% Setup server and wait for client connections
# The server will need to handle multiple clients working in parallel.
# It therefore has to be able to send requrests to multiple clients and not 
# block the program while waiting for a response on previous requests. 
# For this, use the 'select' function which returns if any socket can be read.
# This blocking behavior can be disabled but is kept to keep the program efficient.
t_start = time.time()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))

conns = connect_clients(server, N_MACHINES)
setup_clients(conns, InputList, OutputList)

t_sims = time.time()
print("Conducting multiple runs for setting {}".format(RunList[0]))
states = [0]*N_MACHINES # Array containing simulation counter that every client is running, 0 indicates idle
last_completed = [0]*N_MACHINES # Last completed run number by every client
Results = [0]*(len(RunList)-1)
n_completed = [0]*N_MACHINES # Tracks how many simulations each client has completed

# Loop while the last simulation hasn't been completed or if a client is still running
while max(last_completed) < (len(RunList)-1) or any(states):
    for ID, running in enumerate(states):
        # If this client is not running and the last run is not (being) done
        if not running and max(max(states),max(last_completed)) < len(RunList)-1:
            # Assign the next simulation to this client
            assigned = max(max(states),max(last_completed)) + 1
            Settings[RunList[0]] = RunList[assigned] # Overwrite standard value in settings dict with value in RunList
            conns[ID].sendall(pickle.dumps(Settings)) # Send simulation request to client
            states[ID] = assigned # Designate client as running that specific request
    
    show_state_simulating(states, last_completed)        
    
    # Check if any connection is readable, this indicates returned results or a connection drop
    readable, _, _ = select.select(conns, [], [])
    for conn in readable:
        ID = conns.index(conn)
        received = conn.recv(4096)
        if not received:
            print("Client {} dropped connection".format(ID))
            terminate(server,conns)
        else:
            # These are simulation results
            Results[states[ID]-1] = pickle.loads(received)
            
            # Set last completed and mark this client as idle for next task
            last_completed[ID] = states[ID]
            states[ID] = 0
            n_completed[ID] += 1
            

# All simulations completed, close connection
show_state_simulating(states, last_completed)
print("All simulations completed, plotting and exiting")
x = RunList[1:]
for i in range(0,len(OutputList)):
    y = []
    for ii in range(0,len(Results)):
        y.append(Results[ii][i])
    plt.figure()
    plt.plot(x,y,linewidth=3)
    plt.xlabel(RunList[0])
    plt.ylabel(OutputList[i])
    plt.title("Simulation results for {} for varying {}".format(OutputList[i],RunList[0]))
    plt.grid(True)

t_end = time.time()

print("Total time: {}".format(round(t_end-t_start, 2)))
print("Simulation time: {}".format(round(t_end - t_sims, 2)))
terminate(server,conns)
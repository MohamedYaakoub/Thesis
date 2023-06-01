# -*- coding: utf-8 -*-
"""
Created on Tues Feb 15 2022
Implementation of a parallel GSP method. Server connects to an arbitrary number
of clients and designates simulations to these clients. 
SERVER

@author: BramB
"""
# %%
import socket
import sys
import select
import numpy as np
import pickle
from matplotlib import pyplot as plt
import time


# %% Termination function, called on close
def terminate():
    # Properly shutdown and close server
    for conn in conns:
        conn.close()

    server.close()
    # sys.exit()


# %% Wait for client connection, save connections to conns with position equal to ID
def connect_clients(server, n_machines):

    # State tracker of machine setup
    global state
    state = np.zeros((N_MACHINES, 5))

    conns = [0] * n_machines

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
    print("\033[H\033[J")  # Clears the console
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


    # Send input list to all clients
    for ID, conn in enumerate(conns):
        conn.sendall(pickle.dumps(InputList))

        # Client sends list back to check if connection is solid
        if pickle.loads(conn.recv(4096)) != InputList:
            print("InputList was not properly transferred to client {}".format(ID))
            terminate()
        else:
            state[ID, 1] = 1
            show_state_setup(state)

    # Send output lists to all clients
    for ID, conn in enumerate(conns):
        conn.sendall(pickle.dumps(OutputList))

        # Client sends list back to check if connection is solid
        if pickle.loads(conn.recv(4096)) != OutputList:
            print("OutputList was not properly transferred to client {}".format(ID))
            terminate()
        else:
            state[ID, 2] = 1
            show_state_setup(state)

    # Wait for all the clients to finish loading GSP and giving OK
    while not all(state[:, 4]):
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
                    terminate()
            else:
                # Input/Output count message
                if int(conn.recv(4096)):
                    state[ID, 4] = 1
                    show_state_setup(state)
                else:
                    print("Client {} does not have a matching GSP model, exiting".format(ID))
                    terminate()

    print("\nClient setup succesfully completed")


# %% Prints the state during the simulating process
def show_state_simulating(states, last_completed):
    # print("\033[H\033[J")  # Clears the console
    # print("Parallel-GSP Simulating")
    # print("___________________________\n")

    id_string = ""
    states_string = ""
    last_completed_string = ""
    n_completed_string = ""
    for ID in range(0, len(states)):
        id_string += "  " + str(ID) + "‍  | "
        states_string += " " + "{:03d}".format(states[ID]) + " | "
        last_completed_string += " " + "{:03d}".format(last_completed[ID]) + " | "
        n_completed_string += " " + "{:03d}".format(n_completed[ID]) + " | "

    # print("        Client ID | " + id_string)
    # print("Currently running | " + states_string)
    # print("   Last completed | " + last_completed_string)
    # print(" Amount completed | " + n_completed_string)

def set_up_con(Workers, InputList, OutputList):  # main fucntion to set up the connections between the server and the clients
    # Networking settings
    HOST = ""  # Listen on all adresses
    PORT = 1919  # Port to listen on
    global N_MACHINES
    N_MACHINES = Workers  # Number of clients expected
    # %% Setup server and wait for client connections
    # The server will need to handle multiple clients working in parallel.
    # It therefore has to be able to send requests to multiple clients and not
    # block the program while waiting for a response on previous requests.
    # For this, use the 'select' function which returns if any socket can be read.
    # This blocking behavior can be disabled but is kept to keep the program efficient.
    global t_start
    global server
    global conns
    t_start = time.time()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    conns = connect_clients(server, N_MACHINES)
    setup_clients(conns, InputList, OutputList)


def run(Settings):

    # The number of runs to be conducted
    RunList = len(Settings)  # ["BPR"] + np.linspace(5.1, 5.3, 10).tolist()

    t_sims = time.time()
    print("Conducting multiple runs")
    global n_completed
    global last_completed
    global states

    states = [0] * N_MACHINES  # Array containing simulation counter that every client is running, 0 indicates idle
    last_completed = [0] * N_MACHINES  # Last completed run number by every client
    Results = [0] * RunList
    n_completed = [0] * N_MACHINES  # Tracks how many simulations each client has completed

    # Loop while the last simulation hasn't been completed or if a client is still running
    while max(last_completed) < RunList or any(states):
        for ID, running in enumerate(states):
            # If this client is not running and the last run is not (being) done
            if not running and max(max(states), max(last_completed)) < RunList:
                # Assign the next simulation to this client
                assigned = max(max(states), max(last_completed)) + 1
                Settingsi = Settings[assigned-1]  # The input values assigned from the settings matrix
                conns[ID].sendall(pickle.dumps(Settingsi))  # Send simulation request to client
                states[ID] = assigned  # Designate client as running that specific request

        show_state_simulating(states, last_completed)

        # Check if any connection is readable, this indicates returned results or a connection drop
        readable, _, _ = select.select(conns, [], [])
        for conn in readable:
            ID = conns.index(conn)
            received = conn.recv(4096)
            if not received:
                print("Client {} dropped connection".format(ID))
                terminate()
            else:
                # These are simulation results
                Results[states[ID] - 1] = pickle.loads(received)

                # Set last completed and mark this client as idle for next task
                last_completed[ID] = states[ID]
                states[ID] = 0
                n_completed[ID] += 1

    # All simulations completed, close connection
    show_state_simulating(states, last_completed)
    print("All simulations completed, exiting")
    # x = RunList[1:]
    # for i in range(0, len(OutputList)):
    #     y = []
    #     for ii in range(0, len(Results)):
    #         y.append(Results[ii][i])
    #     plt.figure()
    #     plt.plot(x, y, linewidth=3)
    #     plt.xlabel(RunList[0])
    #     plt.ylabel(OutputList[i])
    #     plt.title("Simulation results for {} for varying {}".format(OutputList[i], RunList[0]))
    #     plt.grid(True)
    #     plt.show()
    t_end = time.time()

    print("Total time: {}".format(round(t_end - t_start, 2)))
    print("Simulation time: {}".format(round(t_end - t_sims, 2)))

    #terminate(server, conns)

    output = []
    # evaluate the objective function
    for i in range(len(Results)):
        # print(Results[i][-1])
        obj = abs(256 - Results[i][-1])
        output.append(obj)

    return output

# # %% Settings
# # User defined settings, free to change
# Settings = [[5.1, 2.7, 1.7, 13], [5.1, 2.7, 1.7, 13.1], [5.1, 2.7, 1.71, 13], [5.11, 2.7, 1.7, 13],
#             [5.1, 2.71, 1.7, 13]]
#
# Th = run(Settings)


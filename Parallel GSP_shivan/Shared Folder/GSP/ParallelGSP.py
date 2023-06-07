# -*- coding: utf-8 -*-
"""
Created on Mon Jan 07 2022
Class implementation of a parallel GSP method. Import this file using:
~~ from ParallelGSP import GSP_Server ~~

This file also presents an example of how to use the class in the main() function
which can be ran by running this file directly.

Version 2.0
@author: BramB
"""

import socket
import select
import pickle
import numpy as np

# %% Class definitions
class GSP_Server:
    # Called when GSP_Server instance is created
    def __init__(self, GSPfileName, N_MACHINES=3, HOST="", PORT=1919, BUFSIZE=4096, PRETTY_PRINT=False):
        self.GSPfileName = GSPfileName
        self.N_MACHINES = N_MACHINES
        self.HOST = HOST
        self.PORT = PORT
        self.BUFSIZE = BUFSIZE
        self.PRETTY_PRINT = PRETTY_PRINT
        self.conns = [0] * N_MACHINES
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.HOST, self.PORT))
        self.setup_state = [[0, 0, 0, 0, 0] for i in range(N_MACHINES)]


    # Propery close all connections and stop execution 
    def terminate(self):
        # Loop over all connections
        for conn in self.conns:
            # If a connection has not been initialized yet, the list entry
            # equals zero. Skip these entries.
            if conn:
                conn.close()

        self.server.close()


    # Wait till all clients are connected and save relevant data to object
    def connect_clients(self):
        if self.PRETTY_PRINT: self.show_setup_state()

        while True:
            self.server.listen(self.N_MACHINES)  # Listen for connection requests
            conn, _ = self.server.accept()  # Accept new requests
            ID = int(conn.recv(self.BUFSIZE))  # Receive ID send by client

            # Check if ID is already connected
            if not self.conns[ID]:
                # ID is new
                self.conns[ID] = conn  # Add connection to conns
                self.setup_state[ID][0] = True  # Mark ID as connected
                if self.PRETTY_PRINT:
                    self.show_setup_state()  # Show new state
                else:
                    print("Client {} connected".format(ID))


            else:
                # Double connection, something wrong in client setup
                print("Double connection with ID {}. Check client run arguments.".format(ID))
                self.terminate()

            if all(self.conns):
                # True if all expected clients connected
                return

    # Pretty print of setup state
    def show_setup_state(self):
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
        for i in self.setup_state:
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

    # Setup client, transmits inputlist and outputlist, starts GSP on clients
    def setup_clients(self, InputList, OutputList):
        # Send inputlist to all clients
        for ID, conn in enumerate(self.conns):
            conn.sendall(pickle.dumps(InputList))

            # Client sends list back to check if connection is solid
            if pickle.loads(conn.recv(self.BUFSIZE)) != InputList:
                # Not properly transferred, terminate
                print("InputList was not properly transferred to client {}".format(ID))
                self.terminate()
            else:
                self.setup_state[ID][1] = True  # Mark inputlist as transferred
                if self.PRETTY_PRINT: self.show_setup_state()

        # Send outputlist to all clients
        for ID, conn in enumerate(self.conns):
            conn.sendall(pickle.dumps(OutputList))

            # Client sends list back to check if connection is solid
            if pickle.loads(conn.recv(self.BUFSIZE)) != OutputList:
                # Not properly transferred, terminate
                print("OutputList was not properly transferred to client {}".format(ID))
                self.terminate()
            else:
                self.setup_state[ID][2] = True  # Mark outputlist as transferred
                if self.PRETTY_PRINT: self.show_setup_state()

        # send the file name of GSP to all clients
        for ID, conn in enumerate(self.conns):
            conn.sendall(pickle.dumps(self.GSPfileName))

        # Clients will automatically start loading GSP after receiving both lists
        # Wait for all clients to finish loading GSP and give an OK
        while not all(el[4] for el in self.setup_state):
            readable, _, _ = select.select(self.conns, [], [])
            for conn in readable:
                ID = self.conns.index(conn)

                # Check if this is a 'I have loaded GSP' or 'I am ready' message
                # If state[ID, 3] is True the client has already loaded GSP
                # This is then therefore an 'OK' or 'Not OK' message
                if self.setup_state[ID][3]:
                    # 'OK' or 'Not OK' message
                    # Client sends a 1 if OK, 0 if Not OK
                    if int(conn.recv(self.BUFSIZE)):
                        # Client send OK, mark as ready
                        self.setup_state[ID][4] = True
                        if self.PRETTY_PRINT: self.show_setup_state()
                    else:
                        # Client send Not OK, terminate
                        print("Client {} does not have a matching GSP model, exiting".format(ID))
                        self.terminate()
                else:
                    # 'Succesfully loaded GSP' or 'Failed loading GSP message'
                    # Client sends 1 if succesful, 0 if failed
                    if int(conn.recv(self.BUFSIZE)):
                        self.setup_state[ID][3] = True
                        if self.PRETTY_PRINT: self.show_setup_state()
                    else:
                        print("Client {} was unable to load GSP, exiting".format(ID))
                        self.terminate()

        # print("\n🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩")
        print("\n" + "\033[0;32m" + "\033[3m" + "Client setup succesfully completed" + "\033[0;0m")

    # Simulate a list of inputs, returns a list of results
    # Position of results in output corresponds to position of input
    def run_simulations(self, Settings):
        # The server will need to handle multiple clients working in parallel.
        # It therefore has to be able to send requests to multiple clients and not 
        # block the program while waiting for a response on previous requests.
        # For this, use the 'select' function which returns if any socket can be read.
        # 'select' blocks untill any client can be read however. As long as every client 
        # is running a simulation when 'select' is called this does not matter. This 
        # blocking behavior can be disabled but is kept for efficiency.
        simulating_state = [-1] * self.N_MACHINES
        last_completed = [-1] * self.N_MACHINES
        n_completed = [0] * self.N_MACHINES
        if isinstance(Settings[0], float):
            Settings = [Settings]
        RunList = len(Settings)
        Results = [0] * len(Settings)

        for con in self.conns:
            con.sendall(pickle.dumps("INIT"))
            continue
        # Loop while last simulation result has not been received
        while not all(Results):
            # Loop over all clients
            for ID, run_num in enumerate(simulating_state):
                # Check if client is not running and if there are still runs that need to be done
                if run_num == -1 and max(max(simulating_state), max(last_completed)) < RunList:
                    # Assign the next available simulation to this client
                    assigned = max(max(simulating_state), max(last_completed)) + 1
                    # -1 removed from index below
                    Settingsi = Settings[assigned]  # The input values assigned from the settings matrix
                    self.conns[ID].sendall(pickle.dumps(Settingsi))  # Send simulation request
                    simulating_state[ID] = assigned

            if self.PRETTY_PRINT: self.show_simulating_state()

            # Check if any connection is readable, indicating a client message
            readable, _, _ = select.select(self.conns, [], [])

            # Loop over readable connections
            for conn in readable:
                ID = self.conns.index(conn)  # Determine ID for this client
                received = conn.recv(self.BUFSIZE)  # Read actual message

                # Check content of message, if empty string client dropped connection
                if received:
                    # These are simulation results
                    # Save results to Results list using saved run number

                    Results[simulating_state[ID]] = pickle.loads(received)

                    # Set last completed, increment counter, mark as inactive
                    last_completed[ID] = simulating_state[ID]
                    n_completed[ID] += 1
                    simulating_state[ID] = -1  # Mark client as idle
                else:
                    # This client dropped connection
                    # Amount of time it would take to implement a proper handler
                    # for this is not worth. Terminate program.
                    print("Client {} dropped connection".format(ID))
                    self.terminate()

        # All simulations completed, return results
        if self.PRETTY_PRINT: self.show_simulating_state()
        # print("All simulations completed")
        return Results

    # Pretty print of simulating state
    def show_simulating_state(self):
        print("\033[H\033[J")  # Clears the console
        print("Parallel-GSP Simulating")
        print("___________________________\n")

        id_string = ""
        states_string = ""
        last_completed_string = ""
        n_completed_string = ""
        for ID in range(0, len(simulating_state)):
            id_string += "  " + str(ID) + "‍  | "
            states_string += " " + "{:03d}".format(simulating_state[ID]) + " | "
            last_completed_string += " " + "{:03d}".format(last_completed[ID]) + " | "
            n_completed_string += " " + "{:03d}".format(n_completed[ID]) + " | "

        print("        Client ID | " + id_string)
        print("Currently running | " + states_string)
        print("   Last completed | " + last_completed_string)
        print(" Amount completed | " + n_completed_string)


# %% Main function, ran if file is ran directly
def main(Settings, InputList, OutputList, RunList):
    MultiGSP = GSP_Server(N_MACHINES=3)  # Initiate GSP_Server instance
    MultiGSP.connect_clients()  # Wait for all clients to connect
    MultiGSP.setup_clients(InputList, OutputList)  # Setup all clients

    # Run all simulations
    Results = MultiGSP.run_simulations(Settings, RunList)

    # Plot results
    x = RunList[1:]
    for i in range(0, len(OutputList)):
        y = []
        for ii in range(0, len(Results)):
            y.append(Results[ii][i])

        plt.figure()
        plt.plot(x, y, linewidth=3)
        plt.xlabel(RunList[0])
        plt.ylabel(OutputList[i])
        plt.title("Simulation results for {} for varying {}".format(OutputList[i], RunList[0]))
        plt.grid(True)

    return MultiGSP, Results




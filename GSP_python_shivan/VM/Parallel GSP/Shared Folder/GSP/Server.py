# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:35:40 2022
New implementation of a parallel GSP server utilizing the new class system.
Serves as an example of how to utilize the class imported from ParallelGSP


@author: BramB
"""

from ParallelGSP import GSP_Server
from matplotlib import pyplot as plt
import numpy as np

# %% Settings
# User defined settings, free to change
Settings = [
     96.96,  # Shaft 1 speed as percentage
    71.22,  # Relative humidity
    285.59,  # Ambient static temperature
    0.9867,  # Ambient static pressure
    0,  0,  # LPC duct (fan+bypass)
    0,  0,  # LPC core (fan+LPC)
    0,  0,
    0,  0,
    0,  0 ]
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


# %% Main program
MultiGSP = GSP_Server(N_MACHINES=1)  # Initiate GSP_Server instance
MultiGSP.connect_clients()  # Wait for all clients to connect
MultiGSP.setup_clients(InputList, OutputList)  # Setup all clients

# Run all simulations
Results = MultiGSP.run_simulations(Settings)

Results2 = MultiGSP.run_simulations(Settings)


# It is good practice to terminate, this also closes all clients and stops
# execution of the python scripts on the clients. Not doing this allows for more
# simulations to be ran using the .run_simulations(Settings, RunList) method.
MultiGSP.terminate()

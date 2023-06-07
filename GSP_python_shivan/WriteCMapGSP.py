"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 5/24/2022 3:28 PM  
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess

# read data
def read_mapC(fileOld, fileNew, map):
    # path  maps for the reference (old) map
    path = "C:/Users/mohsy/University/KLM/Thesis/My thesis/Parallel GSP/Shared " \
           "Folder/GSP/"
    foldername    = "Model-0"
    foldernameRef = "maps_ref"
    path_fileRef  = path + foldername + "/" + foldernameRef + "/" + fileOld + ".map"
    path_fileNew  = path + foldername + "/" + fileNew + ".map"

    path_file = path_fileRef if map == 0 else path_fileNew

    data   = {"list_M": [], "list_E": [], "list_PR": [], "list_S": []}

    data2  = {"list_M": [], "list_E": [], "list_PR": [], "list_S": [], "list_N": []}

    lists  = list(data.keys())
    params = ["Mass Flow", "Efficiency", "Pressure Ratio", "Surge Line"]

    with open(path_file, 'r') as f:
        lines = f.readlines()[2:]
        for line_i, line in enumerate(lines):
            line = line.rstrip()
            if line in params:
                key = lists[params.index(line)]
            if not (line in params) and bool(line):
                data[key].append(line)
        for line_i, line in enumerate(lines):
            line = line.rstrip()
            if line_i != 0 and "1.00" in line:
                lines_beta = line_i
                break

    for param in lists:
        n = 0
        string = data[param]
        # remove beta values
        string = string[lines_beta:] if param != "list_S" else string
        for i, line in enumerate(string):
            words = line.split()
            if i % lines_beta == 0:
                n = words[0]
            if param != "list_S":
                try:
                    if n == words[0] and i % lines_beta == 0: # ToDO last part statement is added
                        words = words[1:]
                except ValueError:
                    pass
            for word in words:
                if param == "list_M":

                        data2["list_N"].append(float(n))
                data2[param].append(float(word))

    Mdot = np.array(data2[lists[0]])
    Eta = np.array(data2[lists[1]])
    PR = np.array(data2[lists[2]])
    surge = np.array(data2[lists[3]])
    surge_m = surge[:int(len(surge) / 2)]
    surge_p = surge[int(len(surge) / 2):]
    N = data2["list_N"]
    return Mdot, Eta, PR, surge_m, surge_p, N

# write data
def write_mapC(fileOld, fileNew, Mdot, Eta, PR, surge_m, surge_p, N):
    # path  maps for the reference (old) and New (gsp) map
    path = "C:/Users/mohsy/University/KLM/Thesis/My thesis/Parallel GSP/Shared " \
           "Folder/GSP/"
    foldername    = "Model-0"
    foldernameRef = "maps_ref"

    path_fileOld  = path + foldername + "/" + foldernameRef + "/" + fileOld + ".map"
    path_fileNew  = path + foldername + "/" + fileNew + ".map"

    # extract the header and beta line data
    with open(path_fileOld, 'r') as f:
        linesO = f.readlines()
        for index, line in enumerate(linesO):
            if line.split()[0] == "Mass":
                length_line = len(line.rstrip())
                startBeta   = index
            if "1.000" in line and index > 2:
                endBeta     = index
                break

    # modify and write the new data
    with open(path_fileNew, 'w') as f:
        for line in linesO[:startBeta]:
            f.write(line)
        # write the data
        for parameter, name in zip([Mdot, Eta, PR], ["Mass Flow", "Efficiency", "Pressure Ratio"]):
            f.write(name + "\n")
            for line in linesO[startBeta+1:endBeta+1]:
                f.write(line)
            for n in np.unique(N):
                values = np.round(parameter[np.where(N == n)], 4)
                values = np.insert(values, 0, n)
                length = 5
                lines = np.array([values[x:x+length] for x in range(0, len(values), length)], dtype=object)
                for line in lines:
                    for word in line:
                        word = str(word)
                        f.write(word.rjust(length_line+4))
                    f.write('\n')
            f.write(" \n")

        f.write("Surge Line\n")
        for it, values in enumerate([surge_m, surge_p]):
            length = 5
            lines = np.array([values[x:x + length] for x in range(0, len(values), length)], dtype=object)
            for line in lines:
                for word in line:
                    word = str(np.round(word, 4))
                    f.write(word.rjust(length_line + 4))
                if line[0] != lines[-1][-1] or it != 1:
                    f.write('\n')

#%%
if __name__ == "__main__":
    path = os.getcwd()
    foldername = "Maps"  # data folder name
    filenameNew = "Fanbcore1"
    filenameOld = "Fanbcore"

    path_fileNew = path + "\\" + foldername + "\\" + filenameNew + ".map"
    path_fileOld = path + "\\" + foldername + "\\" + filenameOld + ".map"
    MdotC, EtaC, PRC, surge_mC, surge_pC, NC = read_mapC("2_LPC_bypass", "2_LPC_bypass", 1)
    plt.scatter(MdotC, PRC, c='grey', marker=".", edgecolors='grey', s=50)
    plt.show()
    plt.scatter(MdotC, EtaC, c='grey', marker=".", edgecolors='grey', s=50)
    plt.show()
    # write_mapC("2_LPC_bypass", "2_LPC_bypass", MdotC, EtaC, PRC, surge_mC, surge_pC, NC)
    # subprocess.Popen([path_fileNew], shell=True)


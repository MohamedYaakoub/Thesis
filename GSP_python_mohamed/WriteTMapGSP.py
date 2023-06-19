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
def read_mapT(fileOld, fileNew, map):
    # path  maps for the reference (old) map
    # path = "C:/Users/mohsy/University/KLM/Thesis/My thesis/Parallel GSP/Shared " \
    #        "Folder/GSP/"
    # foldername    = "Model-0"
    # foldernameRef = "maps_ref"
    path_fileRef = "GEnx Maps Shivan/" + fileOld + ".map"
    path_fileNew = "Solver maps/" + fileNew + ".map"

    path_file = path_fileRef if map == 0 else path_fileNew

    data   = {"list_MinPR": [], "list_MaxPR": [], "list_M": [], "list_E": []}

    data2  = {"list_MinPR": [], "list_MaxPR": [], "list_M": [], "list_E": [], "list_PRN": [], "list_N": [],
                                                                                              "list_B": []}
    lists  = list(data.keys())
    params = ["Min Pressure Ratio", "Max Pressure Ratio", "Mass Flow", "Efficiency"]

    with open(path_file, 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            line = line.rstrip()
            if line in params:
                key = lists[params.index(line)]
            if not (line in params) and bool(line):
                data[key].append(line)

    for param in lists:
        n = 0
        string = data[param]
        # remove beta values
        if not(param == "list_MinPR" or param == "list_MaxPR"):
            for i, line in enumerate(string):
                words = line.split()
                if float(words[0]) == data2["list_PRN"][1]:
                    string = string[i:]
                    lines_beta_values = i
                    break
                if param == "list_M":
                    for word in words: data2["list_B"].append(float(word)) if word != string[0].split()[0] else " "

        for i, line in enumerate(string):
            words = line.split()
            for j, word in enumerate(words):
                if param == "list_MinPR" or param == "list_MaxPR":
                    if i >= len(string)/2:
                        data2[param].append(float(word))
                    elif param == "list_MinPR":
                        data2["list_PRN"].append(float(word))
                elif not(float(word) in data2["list_PRN"] and j == 0 and i % lines_beta_values == 0):
                    data2[param].append(float(word))

                elif param == "list_M":
                    n = word  # if float(word) in data2["list_PRN"] else n
                data2["list_N"].append(float(n)) if n != 0 and not(float(word) in data2["list_PRN"] and j == 0 and
                                                                   i % lines_beta_values == 0) else " "

    PRmin = np.array(data2[lists[0]])
    PRmax = np.array(data2[lists[1]])
    Mdot  = np.array(data2[lists[2]])
    Eta   = np.array(data2[lists[3]])
    NPr   = np.array(data2["list_PRN"])
    N     = np.array(data2["list_N"])
    B     = np.array(data2["list_B"])
    return PRmin, PRmax, Mdot, Eta, NPr, N, B

# write data
def write_mapT(fileOld, fileNew, PRmin, PRmax, Mdot, Eta, N, B):
    # path  maps for the reference (old) and New (gsp) map
    # path = "C:/Users/mohsy/University/KLM/Thesis/My thesis/Parallel GSP/Shared " \
    #        "Folder/GSP/"
    # foldername    = "Model-0"
    # foldernameRef = "maps_ref"

    path_fileRef = "GEnx Maps Shivan/" + fileOld + ".map"
    path_fileNew = "Solver maps/" + fileNew + ".map"

    # extract the header and beta line data
    with open(path_fileRef, 'r') as f:
        linesO = f.readlines()
        for index, line in enumerate(linesO):
            if line.split()[0] == "Min":
                length_line = len(line.split()[0]+line.split()[1]) + 1
                header   = index
                break
        for index, line in enumerate(linesO):
            try: # avoid empty lines
                if line.split()[0] == "Mass":
                    startBeta   = index
                    endBeta     = index + int(np.ceil(len(B)/5))
                    break
            except IndexError:
                pass

    # modify and write the new data
    with open(path_fileNew, 'w') as f:
        for line in linesO[:header]:
            f.write(line)
        # write the data
        for parameter, name in zip([PRmin, PRmax, Mdot, Eta],
                                   ["Min Pressure Ratio", "Max Pressure Ratio", "Mass Flow", "Efficiency"]):
            f.write(name + "\n")
            if name == "Min Pressure Ratio" or name == "Max Pressure Ratio":
                for line in linesO[header+1:header+3]:
                    f.write(line)
                length = 5
                lines = np.array([parameter[x:x + length] for x in range(0, len(parameter), length)], dtype=object)
                for line in lines:
                    for word in line:
                        word = str(np.round(word, 4))
                        f.write(word.rjust(length_line))
                    f.write('\n')
            else:
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
                            f.write(word.rjust(length_line))
                        f.write('\n')
            f.write('\n') if name != "Efficiency" else ""


if __name__ == "__main__":
    path = os.getcwd()
    foldername = "Maps"  # data folder name
    filenameNew = "HPTmap1"
    filenameOld = "HPTmap"

    path_fileNew = path + "\\" + foldername + "\\" + filenameNew + ".map"
    path_fileOld = path + "\\" + foldername + "\\" + filenameOld + ".map"

    # subprocess.Popen([path_fileNew], shell=True)

    PRmin, PRmax, Mdot, Eta, NPr, N, B = read_mapT("5_LPT", "5_LPT", 0)
    # write_mapT("5_LPT", "5_LPT", PRmin*2, PRmax*2, Mdot*1.5, Eta*1.001, N, B)
    # PRmin, PRmax, Mdot, Eta, NPr, N, B = read_mapT("5_LPT", "5_LPT", 1)
    listsE = np.array([Eta[x:x+len(B)] for x in range(0, len(Eta), len(B))])
    listsM = np.array([Mdot[x:x + len(B)] for x in range(0, len(Mdot), len(B))])

    listsPR   = np.zeros(listsE.shape)
    for i in range(len(PRmax)-1):
        p    = np.polyfit([0, 1], [PRmin[i+1], PRmax[i+1]], 1)
        PR   = np.polyval(p, B)
        listsPR[i, 0:len(B)] = PR

    for i in range(len(listsE)):
        plt.plot(listsPR[i], listsE[i])

    plt.xlabel("Corrected Massflow")
    plt.ylabel("Efficiency")
    plt.grid()
    plt.show()


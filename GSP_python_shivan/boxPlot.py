"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 5/30/2022 2:01 PM  
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

path = os.getcwd()
Foldername = "DataDocs"  # data folder name
filename = "CF6-80C2 Test Cell Data"

# Load correlation report data
TestCellDataCF6 = pd.read_excel(path + "\\" + Foldername + "\\" + filename + ".xlsx", skiprows=0, index_col=0)

TestCellDataCF6_bf5 = TestCellDataCF6[TestCellDataCF6["Engine Type"] == "CF6-80C2B5F"]

Ps3_pt2 = TestCellDataCF6_bf5["TOTCR PS3/PT2 TOC"].dropna().values
Ps3_pt2 = Ps3_pt2[Ps3_pt2 > 0]
Ps3_pt2_sim = 31.844181459566073
# %%
fig1, ax1 = plt.subplots()
bp = ax1.boxplot(Ps3_pt2, )
bp['caps'][0].set(color='b', linewidth=2)
bp['caps'][1].set(color='b', linewidth=2)
bp['medians'][0].set(color='orange', linewidth=2)
ax1.scatter(1, Ps3_pt2_sim, c='r')
ax1.set_xticklabels([' '])
ax1.set_ylabel("$Ps_3/Pt_2$")
fig1.show()
ax1.tight_layout()

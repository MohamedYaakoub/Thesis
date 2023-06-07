"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 2/1/2022 2:11 PM
"""

import numpy as np


# Using the paper of Guha for optimum FPR

# OPR = 33.507
# etaFan = 0.905
# etaLPT = 0.9
# etaHPT = 0.8865
# eta_NB = 1  # isentropic efficiency bypass nozzle
# eta_m = 0.99
# Th = 256.087 * 1000
# wdot = 824.63 + 2.67
# M = 0
# gamma = 1.4
# B = 5.1265
# R = 287
# Ta = 289.46
#
# eta_k = etaFan * etaLPT * eta_NB * eta_m
# eta_kC = 289.7 / 409.4  # ~ no match

def FPR(etaFan, etaLPT, B, Th, wdot, Ta, M, eta_m=0.99, eta_NB=1, gamma=1.4, R=287):
    eta_k = etaFan * etaLPT * eta_NB * eta_m
    Fn = Th / wdot
    return (1 + (gamma - 1) / (2 + (gamma - 1) * M ** 2) * (
            (1 + B) ** 2 / (B + 1 / eta_k) ** 2 * (Fn / (np.sqrt(gamma * R * Ta)) + M) - M ** 2)) \
           ** (gamma / (gamma - 1))


# %% Optimum TIT based on a study conducted by Guha
# possible variation due to gamma? 1.33

# etaLPC = 0.905
# etaHPC = 0.8783
# eta_t = (etaLPT * etaHPT) / 1
# eta_c = (etaLPC * etaHPC) / 1


# FPR = 1.7265

def FT04(OPR, eta_t, eta_c, Th, wdot, Ta, M, B, gamma=1.4, R=287):
    Fn = Th / wdot

    a = (1 - OPR ** ((1 - gamma) / gamma)) * (eta_t / Ta)

    b = (gamma - 1) * 0.5 * (1 + B)

    c = (eta_t * (Fn / (np.sqrt(gamma * R * Ta)) + M) ** 2 - eta_c * M ** 2)

    d = (OPR ** ((gamma - 1) / gamma) - 1) / eta_c

    T04 = (b * c + d) / a

    Fn = (Fn / 2.204623) * 0.2248089431

    T04 = T04 + 5 * (100 / Fn - B)

    return T04


# %%
# far = 0.01997
# gammaT = 1.3
# href = 297.4408
# ha = 289.46 * 1012.13
# h04 = 1501.49 * 1269.79
# eta_d = 1
# eta_n = 0.9
# eta_fn = 0.915
# PRC = OPR / FPR
# PRB = 0.96
# m = 1 + (gamma - 1) / 2 * M ** 2

# far    = 0.023244
# gammaT = 1.3
# href   = 297.4408
# ha     = 285.59*1009.75
# h04    = 1290.95*1659.13
# eta_d  = 1
# eta_n  = 0.9
# eta_fn = 0.915
# FPR = 1.5482
# OPR = 46.268882
# PRC = OPR / FPR
# PRB    = 0.96
# m      = 1 + (gamma-1)/2 * M**2
# eta_c  = (0.872*0.900)/1
# eta_t  = (0.930*0.940)/1
# etaFan = 0.92

# far    = 0.02
# gammaT = 1.350
# href   = 297.4408
# ha     = 200
# h04    = 1000
# eta_d  = 0.82
# eta_n  = 0.95
# eta_fn = 0.95
# FPR    = 2.25
#
# PRC    = 11.1
# PRB    = 0.97
# m      = 1 + (gamma-1)/2 * M**2
# eta_c  = 0.86
# eta_t  = 0.92
# etaFan = 0.8

# A = (1 + far) * np.sqrt(h04 / ha * eta_n)
# a = (1 / (1 + eta_d * (m - 1))) ** (((gammaT - 1) / (gamma - 1)) * (gamma / gammaT))
# b = (1 / (PRC * PRB)) ** ((gammaT - 1) / gammaT)
# c = (m / (eta_t * (1 + far)) * (ha / h04) * (PRC ** ((gamma - 1) / gamma) - 1) / eta_c)
# B = 1 - c - a * b
# C = m / (eta_t * (1 + far)) * (ha / h04) * (FPR ** ((gamma - 1) / gamma) - 1) / eta_t
# D = np.sqrt(m * eta_fn) * np.sqrt((1 + (FPR ** ((gamma - 1) / gamma) - 1) / etaFan) * (1 - 1 / ((1 + eta_d * (m - 1))
#                                                                                                 * FPR ** ((
#                                                                                                                       gamma - 1) / gamma))))
#
# BPR = B / C - A ** 2 * C / (4 * (D - np.sqrt((gamma - 1) / 2) * M) ** 2)

# BPR      = 5.1265
#
# T = (A*(B-BPR*C)**0.5 + BPR*D - (1+BPR)*M * (0.5 * (gamma-1))**0.5)

# tr_r = 1 + 0.2 * M ** 2  # ram
# tr_f = FPR ** (0.4 / 1.4)  # fan
# tr_f2 = FPR ** (0.4 * 0.8 / 1.4)  # fan
# tr_c = PRC ** (0.4 / 1.4)  # compressor
# Hr = 1659 / 285
# Hr = 1501 / 289
# tr_r = 1 + 0.2*0.9**2  # ram
# tr_f = 2**(0.4/1.4)  # fan
# tr_c = 15**(0.4/1.4)  # compressor
# Hr = 1670/217


# BPR = 1 / (tr_r * (tr_f - 1)) * (
#         Hr - tr_r * (tr_c - 1) - (Hr / (tr_r * tr_c)) - 0.25 * (np.sqrt(tr_r * tr_f - 1) + np.sqrt(tr_r - 1)) ** 2)
#
# TR = 1 + 1 / 0.92 * (tr_f - 1)

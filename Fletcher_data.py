"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 1/14/2022 4:00 PM  
"""

import numpy as np
import matplotlib.pyplot as plt


# %% data
compr = [[0.19963, 0.25087, 0.30064, 0.35114, 0.40127, 0.45067, 0.50044, 0.57032, 0.59996, 0.64972, 0.69837],
         [0.91982, 0.91573, 0.91073, 0.90482, 0.89818, 0.89082, 0.88236, 0.86933, 0.86309, 0.85246, 0.84055]]

fan = [[0.4003662214200535, 0.45020434361736855, 0.5005971103744699, 0.550985169915375, 0.6013678497806275,
        0.6514664827714212, 0.7004410325345978, 0.7499715717764737, 0.8001970978667237],
       [0.909307487802426, 0.9084335460434672, 0.9064829697961473, 0.9035909973817791, 0.8996231436336414,
        0.8948487554624722, 0.8892689626001586, 0.8828815055829266, 0.8754844683720463]]

z_comp = np.polyfit(compr[0], compr[1], 3)
f_comp = np.poly1d(z_comp)

z_fan = np.polyfit(fan[0], fan[1], 3)
f_fan = np.poly1d(z_fan)

def loading_test(rmp, stages, r_mean, dH):
    Um = 2*np.pi * r_mean  * rmp/60
    if stages != 1:
        Um = 300
    else:
        if r_mean == 1:
            Um = 310
        else:
            Um = 430

    return dH/(stages*Um**2)

def loading(rmp, stages, r_mean, dH):
    Um = 2*np.pi * r_mean  * rmp/60
    return dH*1000/(stages*Um**2)

def flow_coef(rmp, r_mean, Vx):
    Um = 2 * np.pi * r_mean * rmp / 60
    return Vx/Um


def etaP_C(loading):
    return f_comp(loading)


def etaP_f(loading):
    return f_fan(loading)

# %%
def eta_isC(eta_p, PRc, gamma):
    return (PRc**((gamma - 1)/gamma) - 1) / (PRc**((gamma - 1)/(gamma * eta_p)) - 1)

def eta_isT(eta_p, PRt, gamma):
    return (1 - PRt**(eta_p * (1 - gamma)/gamma))/(1 - PRt**((1 - gamma)/gamma))

def eta_thermal(m_dot, BPR, vjc, vjbp, v0, mf, LHV):
    m_dotBP = BPR*m_dot/(1+BPR)
    m_dotC  = m_dot/(1+BPR)
    # print(m_dotC, m_dotBP, m_dot)
    return 0.5 * (m_dotBP * (vjbp**2 - v0**2) + m_dotC * (vjc**2 - v0**2)) / (mf*LHV)

def eta_prop(m_dot, BPR, vjc, vjbp, v0):
    m_dotBP = BPR*m_dot/(1+BPR)
    m_dotC  = m_dot/(1+BPR)
    return (m_dotBP * (vjbp - v0) + m_dotC * (vjc - v0)) * v0 / (0.5 * (m_dotBP * (vjbp**2 - v0**2) + m_dotC
                                                                        * (vjc**2 - v0**2)))


#%%
if __name__=="__main__":
    plt.rcParams['figure.dpi'] = 500
    fig, ax = plt.subplots()
    loading_list = np.linspace(0.2, 0.8, 25)
    ax.plot(loading_list, f_comp(loading_list), color='k', linestyle="-", label='Reference data')
    ax.scatter(loading_list, f_comp(loading_list), marker='*', color='r')
    ax.set_xlabel('Average loading ' + r'$\Psi$', fontsize=18)
    ax.set_ylabel(r'$\eta_p$', fontsize=18)
    ax.set_title('Polytropic efficiency for Compressor', fontsize=20)
    ax.grid()
    fig.show()

    fig1, ax1 = plt.subplots()
    ax1.plot(loading_list, f_fan(loading_list), color='k', linestyle="-", label='Reference data')
    ax1.scatter(loading_list, f_fan(loading_list), marker='*', color='r')
    ax1.set_xlabel('Average loading ' + r'$\Psi$', fontsize=18)
    ax1.set_ylabel(r'$\eta_p$', fontsize=18)
    ax1.set_title('Polytropic efficiency for fan', fontsize=20)
    ax1.grid()
    fig1.show()


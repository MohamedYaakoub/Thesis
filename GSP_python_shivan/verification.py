"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 5/4/2022 1:42 PM  
"""
import numpy as np
import matplotlib.pyplot as plt
from Fletcher_data import loading, f_comp, f_fan, eta_prop, eta_thermal, flow_coef
from parameters import params, params2
from smithChart import xi, yi, zi, x, y, rbf

plt.rcParams['figure.dpi'] = 500
plt.figure()
# fig for HPC and Fan core
fig, ax = plt.subplots()
plt.figure()
# fig for HPC
fig1, ax1 = plt.subplots()
plt.figure()
# fig for LPT and HPT
fig2, ax2 = plt.subplots()
# sp = ax2.scatter(xi, yi, c=zi, cmap='jet')
# ax2.scatter(x, y, c='k', marker="+", s=30)
# cb = fig2.colorbar(sp)
plt.figure()


mach = 0.25
v0 = 85  # m/s

# %%
for i in [0, 1]:
    Engine = i
    p = params() if Engine == 0 else params2()
    loading_list = np.linspace(0.2, 0.8, 25)
    if Engine == 0:
        inputs_list = ["BPR", "FPRbp", "HPCpr", 'HPC_ef', 'fan_etaC', 'HPT_ef', 'LPT_ef', 'fan_etaD', 'Cx_core',
                       'Cx_bp']
        output_list = ["TT25", "Ps14", "TT3", "Ps3", "Pt49", "TT49", "T5", "FN", "dH_HPC", "dH_FanC", "dH_FanBp",
                       "A_c", "A_bp", "Eta_fan", "VxHPTin", "VxHPTout", "VxLPTout", "dH_HPT", "dH_LPT"]

        X = [5.084086470704756, 1.7280973721923745, 12.97756833279673, 0.8973505330918321, 0.903470456888131,
             0.9082592265792493, 0.8719695048669192, 0.9181738420687525, 0.9188437011863649, 0.9270913176702706]

        outp = [389.076155529274, 1.4456863664859, 847.111754567586, 32.2918687336931, 7.64363294563538,
                1142.08883737004, 827.552588096995, 257.752200008003, 488.666174502788, 98.445365699532,
                54.7989375302539, 0.626050453918499, 1.80176085874956, 0.908229404469543, 127.54311416247,
                251.449707309, 239.610172143396, -483.635883982646, -373.16645626723]
        vjc = 463.6
        vjbp = 318.4
        LHV = 43031 * 1000
    else:
        inputs_list = ["BPR", "FPRc", "FPRbp", "HPCpr", 'HPC_ef', 'fan_etaC', 'HPT_ef', 'LPT_ef', 'fan_etaD', 'Cx_core',
                       'Cx_bp']
        output_list = ["TT25", "TT3", "Ps3", "TT49", "FN", "dH_HPC", "dH_FanC", "dH_FanBp",
                       "A_c", "A_bp", "Eta_fan", "VxHPTin", "VxHPTout", "VxLPTout", "dH_HPT", "dH_LPT"]
        X    = [9.099999609651045, 2.1549919990126183, 1.5752182041381115, 21.391284427002535, 0.9101106864202795,
         0.8802054162523129, 0.8971994049002746, 0.8969494640950004, 0.8987228185093691, 0.9306593868837868,
         0.9450679400258979]
        outp = [353.394349337384, 886.596388678527, 45.3500593761777, 1188.33612903657, 321.669938366132,
                 503.252511423, 101.272019377, 54.3118324654, 0.669403188462468, 3.06434924572457, 0.889520701461887,
                 94.8863331588892, 203.294684457663, 174.992151723068, -558.462075768137, -464.728332773172]
        vjc = 379.4
        vjbp = 279.1
        LHV = 43031 * 1000


    # %%
    def eta_HPC(x):

        return x[inputs_list.index('HPC_ef')], loading(p.inputs['N2'], p.stagesHPC, p.RmHPC,
                                                       outp[output_list.index("dH_HPC")])


    def eta_fanC(x):

        return x[inputs_list.index('fan_etaC')], loading(p.inputs['N1'], p.stagesB, p.RmfanC,
                                                         outp[output_list.index("dH_FanC")])


    def eta_fanD(x):

        return x[inputs_list.index('fan_etaD')], loading(p.inputs['N1'], 1, p.RmfanBp,
                                                         outp[output_list.index("dH_FanBp")])


    def eta_Hpt(x):

        flowCin = flow_coef(p.inputs['N2'], p.RmHPTin, outp[output_list.index("VxHPTin")])
        flowCout = flow_coef(p.inputs['N2'], p.RmHPTout, outp[output_list.index("VxHPTout")])
        flowC = 0.5 * (flowCout + flowCin)
        lding = loading(p.inputs['N2'], p.stagesHPT, 0.5 * (p.RmHPTin + p.RmHPTout), -outp[output_list.index("dH_HPT")])
        print(rbf(flowC, lding))
        return flowC, lding


    def eta_Lpt(x):

        V_LPT = 0.5 * (outp[output_list.index("VxHPTout")] + outp[output_list.index("VxLPTout")])
        flowC = flow_coef(p.inputs['N1'], p.RmLPT, V_LPT)

        lding = loading(p.inputs['N1'], p.stagesLPT, p.RmLPT, -outp[output_list.index("dH_LPT")])
        print(rbf(flowC, lding))
        return flowC, lding


    e_HPC = eta_HPC(X)
    e_fanc = eta_fanC(X)
    e_fand = eta_fanD(X)
    e_Hpt = eta_Hpt(X)
    e_Lpt = eta_Lpt(X)
    print(e_Hpt)

    # %% efficiencies
    eta_TH = eta_thermal(p.inputs['dot_w'], X[inputs_list.index('BPR')], vjc, vjbp, v0, p.inputs['Wf'], LHV)
    eta_pr = eta_prop(p.inputs['dot_w'], X[inputs_list.index('BPR')], vjc, vjbp, v0)

    print('_________________Engine: {}__________________'.format('CF6' if Engine == 0 else 'GEnx'))
    print('Eta_th:', eta_TH)
    print('Eta_p :', eta_pr)
    print('\n')
    # %% plot


    ax.scatter(e_HPC[1], e_HPC[0], marker='o',
               color='blue' if Engine == 0 else 'orange',
               label='CF6 HPC' if Engine == 0 else 'GEnx HPC',
               linewidths=3, s=60)
    ax.scatter(e_fanc[1], e_fanc[0], marker='x',
               color='blue' if Engine == 0 else 'orange',
               label='CF6 Fan Core' if Engine == 0 else 'GEnx Fan Core',
               linewidths=3, s=60)
    ax1.scatter(e_fand[1], e_fand[0], marker='o' if Engine == 0 else 'o',
                color='blue' if Engine == 0 else 'orange',
                label=' CF6 Fan Duct' if Engine == 0 else 'GEnx Fan Duct',
                linewidths=3, s=60)
    ax2.scatter(e_Lpt[0], e_Lpt[1], marker='o',
                 color='blue' if Engine == 0 else 'orange',
                 label='CF6 LPT' if Engine == 0 else 'GEnx LPT',
                linewidths=3, s=60)
    ax2.scatter(e_Hpt[0], e_Hpt[1], marker='x',
                 color='blue' if Engine == 0 else 'orange',
                 label='CF6 HPT' if Engine == 0 else 'GEnx HPT',
                linewidths=3, s=60)

ax.plot(loading_list, f_comp(loading_list), color='k', linestyle="--", label='Reference data')
ax.set_xlabel('Average loading ' + r'$\Psi$', fontsize=16)
ax.set_ylabel(r'$\eta_p$', fontsize=16)
       #title='Polytropic efficiency for Compressor')
ax.grid()
ax.legend()
fig.show()

ax1.plot(loading_list, f_fan(loading_list), color='k', linestyle="--", label='Reference data')
ax1.set_xlabel('Average loading ' + r'$\Psi$', fontsize=16)
ax1.set_ylabel(r'$\eta_p$', fontsize=16)
        #title='Polytropic efficiency for fan')
ax1.grid()
ax1.legend()
fig1.show()

# cb.set_label('Efficiency', loc='center')
# ax2.set(xlabel=r"Flow Coefficient ($\phi$)", ylabel=r"Loading Coefficient ($\Psi$)",)# title="Smith chart")
CS = ax2.contour(xi, yi, zi, levels=[0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94], cmap="twilight_shifted")
ax.clabel(CS, inline=True, fontsize=10)
ax2.set_xlabel(r"Flow Coefficient ($\phi$)", fontsize=16)
ax2.set_ylabel(r"Loading Coefficient ($\Psi$)", fontsize=16)
plt.grid()
ax2.legend()
fig2.show()

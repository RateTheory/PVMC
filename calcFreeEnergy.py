#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:03:03 2020
Free energy calculations
@author: Shaama M. Sharada
"""

# Inputs
# 1. Temperature
# 2. (Exact/approximate) eigenvalues
# 3. Electronic energies


import numpy as np
import math
import matplotlib

# import bigfloat

font = {"family": "sans-serif", "weight": "normal", "size": 15}

matplotlib.rc("font", **font)


Tarr = [0.001, 200.0, 298.15, 500.0, 1000.0]  # temperature

h = 6.62617353e-34
c = 2.99792458e10
Na = 6.02204531e23
kcaltokJ = 4.184
kb = 1.3806244e-23
R = 8.3144126
amu = 1.6605402e-27
htokcal = 627.5096


def ZPE_calc(cm):
    # assuming that all frequencies are real
    # cm is an array
    return np.sum(cm) * 0.5 * h * c * Na / (1e3 * kcaltokJ)


def free_energy_vtst(E, cm, T, cm_true=None, vmc=False):
    Hovrkt = h / (kb * T)
    G = 0
    j = 0
    use_cutoff = True
    for freq in cm:
        if (
            vmc
        ):  # If we want the same number of modes included from Xfinal and Xtrue. Deemed deprecated 5/1/2021
            if cm_true[j] != 0 and cm_true[j] >= freq_cutoff and freq != 0:
                frqsi = np.longdouble(freq * c)
                G += R * T * np.log(1 - np.exp(-Hovrkt * frqsi))
            if (
                cm_true[j] < freq_cutoff and use_cutoff
            ):  # If less than cutoff, replace with cutoff
                frqsi = np.longdouble(freq_cutoff * c)
                G += R * T * np.log(1 - np.exp(-Hovrkt * frqsi))
                # print('ping')
            j = j + 1
        else:
            if freq != 0 and freq >= freq_cutoff:
                frqsi = np.longdouble(freq * c)
                G += R * T * np.log(1 - np.exp(-Hovrkt * frqsi))

            if (
                freq < freq_cutoff and use_cutoff
            ):  # If less than cutoff, replace with cutoff
                frqsi = np.longdouble(freq_cutoff * c)
                G += R * T * np.log(1 - np.exp(-Hovrkt * frqsi))
                # print('ping')
        # elif freq!=0 and freq<freq_cutoff: #SMS 0520 - makes things worse
        #    frqsi = np.longdouble(freq_cutoff*c)
        #    G+= R*T*np.log(1-np.exp(-Hovrkt*frqsi))
    G *= 1.0 / (kcaltokJ * 1000.0)
    G += E * htokcal
    G += ZPE_calc(cm)
    # if T==1000:
    # print(G)
    return G


def enthalpy_entropy(cm, T):
    H = 0
    S = 0
    Hovrkt = h / (kb * T)
    for freq in cm:
        if freq != 0:
            frqsi = np.longdouble(freq * c)
            ui = frqsi * Hovrkt
            expui = np.longdouble(np.exp(ui))
            # print(freq, frqsi/(expui-1.))
            H += frqsi / (expui - 1.0)
            S += ui / (expui - 1) - np.log(1 - 1.0 / expui)
        else:
            H += 0.0
            S += 0.0
    H *= h * Na / (kcaltokJ * 1000.0)
    H += ZPE_calc(cm)
    S *= R / kcaltokJ
    return H, S


def free_energy(E, H, S, T):
    return E * htokcal + H - (T * S / 1.0e3)


def au_to_wvno(eigval):
    # converts from atomic units to wavenumbers
    # copying formula from qchem trunk for now
    tocm = 5.89141e-7
    tocm2 = 3.94562
    if eigval > 0.0:
        return np.sqrt(math.fabs(eigval) / tocm) * tocm2
    else:
        # print("Negative eigenvalue, ", eigval)
        return 0.0


freq_cutoff = 100  # cm-1

# cutoff = 1.895e-4 #50cm-1; 3.79e-4 #=100.075cm-1; 8.52e-4 #150.05cm-1


"""
df = pd.read_pickle('Sn2Ar2_ordered_dft2svp_step100_disp5000_num100.pkl')

plotall = False

for index, row in df.iterrows():
    #print(index)
    evals = row.Eigenvalues
    cm = np.array([au_to_wvno(x) for x in evals])

    #print(cm)
    zpe = ZPE_calc(cm)
    df.loc[index,'ZPE']=zpe
    for T in Tarr:
        enthalpy, entropy = enthalpy_entropy(cm,T)
        colnameH = 'Enthalpy, '+str(int(T))+'K'
        colnameS = 'Entropy, '+str(int(T))+'K'
        colnameG = 'Free energy, '+ str(int(T))+'K'

        df.loc[index,colnameH] = enthalpy
        df.loc[index,colnameS] = entropy
        #df.loc[index,colnameG] = free_energy(row['Energy'],enthalpy, entropy, T)
        df.loc[index,colnameG] = free_energy_vtst(row.Energy,cm,T)


if plotall:
    fig, ax1 = plt.subplots()
    y1 = (df['Energy']-df.loc[min(df.index)]['Energy'])*htokcal
    y2 = (df['Energy']-df.loc[min(df.index)]['Energy'])*htokcal+df['ZPE']
    y3 = df['Free energy, 1000K']-df.loc[max(df.index)]['Free energy, 1000K']
    #y1 = y1[:22]
    #y2 = y2[:22]
    #y3 = y3[:22]
    color = 'tab:red'
    ax1.set_xlabel('s')
    ax1.set_ylabel(r'$V_{0}$, $V_{a}^{G}$')
    ax1.plot(df.s,y1,label = r'$V_{0}$',color=color,lw=2.)
    ax1.plot(df.s,y2,label=r'$V_{a}^{G}$',color='blue',lw=2.)
    ax1.legend(loc=3)
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\Delta G_{vib}$')
    ax2.plot(df.s,y3,label=r'$\Delta G_{vib}$, 1000K',color='black',lw=2.)
    ax2.legend(loc=1)
    #plt.grid(axis='x')

"""
# Plots
# T = '500K'
"""
y1 = (df['Energy']-df.loc[min(df.index)]['Energy'])*htokcal
y2 = (df['Energy']-df.loc[min(df.index)]['Energy'])*htokcal+df['ZPE']
y3 = df['Free energy, 1000K']-df.loc[min(df.index)]['Free energy, 1000K']

plt.plot(df.s,y1,marker='', color='skyblue', linewidth=2,label='MEP')
plt.plot(df.s,y2,marker='', color='olive', linewidth=2,label=r'$V_{ag}$')
plt.plot(df.s,y3,marker='', color='red', linewidth=2,label=r'$\Delta G$, 1000K')
plt.legend()
plt.grid()
"""

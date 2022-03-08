#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:49:01 2020
Analyze results
@author: Selin Bac, Stephen Quiton, Shaama M. Sharada
"""

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib, glob
import seaborn as sns
from numpy.linalg import norm
from calcFreeEnergy import ZPE_calc, enthalpy_entropy, au_to_wvno
from calcFreeEnergy import free_energy_vtst
import numpy as np
from matplotlib.lines import Line2D
from scipy.io import loadmat
import ZCT

T_tunnel = [300]  # ZCT Temperature, K.


def get_eigenvalue_matrix(df, tag="All"):
    evals = []
    energies = []
    if tag in ["R", "P", "T"]:
        for index, row in df.iterrows():
            if row.tag == tag:
                evalrow = []
                for x in row.Eigenvalues:
                    # if abs(x)>cutoff:
                    evalrow.append(x)
                evals.append(evalrow)
                energies.append(row.Energy)
    elif tag == "All":
        for index, row in df.iterrows():
            evalrow = []
            for x in row.Eigenvalues:
                # if abs(x)>cutoff:
                evalrow.append(x)
            evals.append(evalrow)
            energies.append(row.Energy)
    elif tag == "Rhalf":
        for index, row in df.iterrows():
            if index <= 0:
                evalrow = []
                for x in row.Eigenvalues:
                    # if abs(x)>cutoff:
                    evalrow.append(x)
                evals.append(evalrow)
                energies.append(row.Energy)
    elif tag == "Phalf":
        for index, row in df.iterrows():
            if index >= 0:
                evalrow = []
                for x in row.Eigenvalues:
                    # if abs(x)>cutoff:
                    evalrow.append(x)
                evals.append(evalrow)
                energies.append(row.Energy)
    return evals, energies


def count_acc(dfcol, acc):
    count = 0
    for val in dfcol:
        if abs(val) <= acc:
            count += 1
    return count


x = loadmat("results.mat")
query_system = x["system"][0]
filename_true = glob.glob("MatrixPKLs/*" + query_system + "*.pkl")
alldf_zct = pd.read_pickle(filename_true[0])
mirror = False  # true if using half of a symmetric reaction, e.g. Sn2
if "Sn2" in filename_true[0]:
    mirror = True
if mirror:
    df_copy = alldf_zct.copy()
    df_copy = df_copy.drop(0)  # Try 0 first. If error, use 1
    df_copy.index = -1 * df_copy.index
    df_copy["s"] = -1 * df_copy["s"]
    df_copy = df_copy.sort_index(ascending=True)
    alldf_zct = df_copy.append(alldf_zct)
s = alldf_zct["s"]
s_list = []
for i in s.index:
    s_list.append(s[i])
evals = alldf_zct["Eigenvalues"]
Vmep_zct = alldf_zct["Energy"]
index_zct = alldf_zct.index

ZCT_true, E0_true, VAG_true, SAG_true, V_aG_true = ZCT.zct(
    T_tunnel, evals, Vmep_zct, s, "calc", "calc", "calc"
)
print("True ZCT:", ZCT_true)

Trange = [100, 200, 500, 1000]

font = {"family": "sans-serif", "weight": "normal", "size": 15}

matplotlib.rc("font", **font)

system = query_system
accuracy = 2.39  # kcal/mol = 10kJ/mol - chemical accuracy

# index corresponding to starting pt/reactant. For half reactions, it is -1, for
# full reactions it is 0
indreact = -1

Xinit = x["Xinit_c"]
Xfinal = x["Xvmc2"]  # Xvmc2 for QS based matrix
Xtrue = x["Xtrue"]
Energies = list(alldf_zct["Energy"])
ntrials = 1
alldf = pd.DataFrame(
    {
        "Trial": [0],
        "Energies": [Energies],
        "Xfinal": [Xfinal],
        "Xtrue": [Xtrue],
        "Xinit": [Xinit],
    }
)  # Empty Dataframe,
#######
# THERMO
#######

# ZPE
ZPE_true = []
for column in Xtrue.T:
    cm = np.array([au_to_wvno(x) for x in column])  # if abs(x)>cutoff])
    zpe = ZPE_calc(cm)
    ZPE_true.append(zpe)
ZPE_true = np.array(ZPE_true)

# Free Energy
G_true = []
cm_true = []
for index, column in enumerate(Xtrue.T):
    cm = np.array([au_to_wvno(x) for x in column])  # if abs(x)>cutoff])

    freeEgyTemp = []
    for T in Trange:
        H, S = enthalpy_entropy(cm, T)
        # print(index,T,Energies)
        # freeEgyTemp.append(free_energy(Energies[index],H,S,T))
        freeEgyTemp.append(free_energy_vtst(Energies[index], cm, T))
    G_true.append(freeEgyTemp)
    cm_true.append(cm)
G_true = np.array(G_true).T

transpose = True
thermodict = []
error_init = []
error_final = []
all_colerr_init = []
all_colerr_vmc = []
ZPE_all = []


labels = []
for Tindex, T in enumerate(Trange):
    label = "G, True, " + str(T) + "K"
    labels.append(label)

G_true_dict = {
    labels[0]: G_true[0, :],
    labels[1]: G_true[1, :],
    labels[2]: G_true[2, :],
    labels[3]: G_true[3, :],
}
G_true_df = pd.DataFrame(G_true_dict)
for index, row in alldf.iterrows():

    Xfinal = row["Xfinal"]
    Xinit = row["Xinit"]

    # Matrix errors
    materror_init = norm(np.subtract(Xinit, Xtrue)) / norm(Xtrue)
    materror_final = norm(np.subtract(Xfinal, Xtrue)) / norm(Xtrue)

    error_init.append(materror_init)
    error_final.append(materror_final)

    ZPE_final = []
    for column in Xfinal.T:
        cm = np.array([au_to_wvno(x) for x in column])  # if abs(x)>cutoff])
        zpe = ZPE_calc(cm)
        ZPE_final.append(zpe)

    ZPE_error = np.subtract(ZPE_final, ZPE_true)
    ZPE_all.append(ZPE_final)

    G_final = []
    # Free energy, VMC
    i = 0
    for subindex, column in enumerate(Xfinal.T):
        cm = np.array([au_to_wvno(x) for x in column])  # if abs(x)>cutoff])

        freeEgyTemp = []

        for T in Trange:
            H, S = enthalpy_entropy(cm, T)
            # freeEgyTemp.append(free_energy(Energies[subindex],H,S,T))
            # freeEgyTemp.append(free_energy_vtst(row['Energies'][subindex],cm,T,cm_true[i],True))
            freeEgyTemp.append(free_energy_vtst(row["Energies"][subindex], cm, T))

        G_final.append(freeEgyTemp)
        i = i + 1

    G_final = np.array(G_final).T

    G_error = np.subtract(G_final, G_true)

    labels = []
    error_labels = []
    for Tindex, T in enumerate(Trange):
        label = "G, VMC, " + str(T) + "K"
        error_label = "G error, " + str(T) + "K"
        labels.append(label)
        error_labels.append(error_label)
        # alldf.loc[index,label]= DeltG_final[Tindex,:]

    thermodict.append(
        {
            "Trial": index,
            "ZPE, VMC": ZPE_final,
            "ZPE error": ZPE_error,
            labels[0]: G_final[0, :],
            labels[1]: G_final[1, :],
            labels[2]: G_final[2, :],
            labels[3]: G_final[3, :],
            error_labels[0]: G_error[0, :],
            error_labels[1]: G_error[1, :],
            error_labels[2]: G_error[2, :],
            error_labels[3]: G_error[3, :],
        }
    )

    # Column errors
    colerr_init = []
    colerr_vmc = []
    errortol = 1.0e-3  # 1e-3
    if transpose:
        for j in range(Xtrue.shape[1]):
            colerr_init.append(norm(Xinit[:, j] - Xtrue[:, j]) / norm(Xtrue[:, j]))
            colerr_vmc.append(norm(Xfinal[:, j] - Xtrue[:, j]) / norm(Xtrue[:, j]))
    else:
        for j in range(Xtrue.shape[0]):
            colerr_init.append(norm(Xinit[j, :] - Xtrue[j, :]) / norm(Xtrue[j, :]))
            colerr_vmc.append(norm(Xfinal[j, :] - Xtrue[j, :]) / norm(Xtrue[j, :]))

    colerr_init.sort()
    colerr_init = colerr_init[::-1]

    count_init = sum(map(lambda x: x <= errortol, colerr_init))

    colerr_vmc.sort()
    colerr_vmc = colerr_vmc[::-1]
    count_vmc = sum(map(lambda x: x <= errortol, colerr_vmc))

    all_colerr_init.append(colerr_init)
    all_colerr_vmc.append(colerr_vmc)

# raise Exception("This is just to get out, comment me when unneeded")
thermodf = pd.DataFrame(thermodict)

for column in thermodf.columns.values:
    if column != "Trial":
        alldf[column] = thermodf[column].values

for index, row in alldf.iterrows():
    maxindzpe = abs(row["ZPE error"]).argmax()
    alldf.loc[index, "ZPE error, max"] = row["ZPE error"][maxindzpe]
    alldf.loc[index, "ZPE error, mean"] = row["ZPE error"].mean()
    for Tindex, T in enumerate(Trange):
        stringlabmax = "G error, max, " + str(T) + "K"
        stringlabelmean = "G error, mean, " + str(T) + "K"
        maxindg = abs(row[error_labels[Tindex]]).argmax()

        alldf.loc[index, stringlabmax] = row[error_labels[Tindex]][maxindg]
        alldf.loc[index, stringlabelmean] = row[error_labels[Tindex]].mean()


# Error vs. Column plot
if transpose:
    cols = np.arange(Xtrue.shape[1])
else:
    cols = np.arange(Xtrue.shape[0])

fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
ax.set_yscale("log")
ax.set_ylim([1.0e-8, 1.0e2])
for i in range(ntrials):
    ax.plot(
        cols, all_colerr_init[i], linestyle="solid", color="r", label="Initial", lw=1.75
    )
    ax.plot(
        cols, all_colerr_vmc[i], linestyle="dashed", color="b", label="HVMC", lw=1.75
    )
    if i == 0:
        ax.legend()
        ax.set_xlabel("Columns")
        ax.set_ylabel("Error")
        ax.grid()
        # anntext = r'$\rho$ = '+str(density)+', '+str(ntrials)+' trials'
        anntext = str(ntrials) + " trials"
        ax.annotate(anntext, (0, 1.0e1), fontsize=13)

# Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
errors = sns.jointplot(
    x=alldf["ZPE error, max"],
    y=alldf["G error, max, 1000K"],
    kind="scatter",
    color="m",
    s=100,
)

errors.set_axis_labels("Max. deviation, ZPE", r"Max. deviation, $G_{vib}$, 1000K")
# errors.set_axis_labels("ZPE",r"$G_{vib}$, 1000K")
plt.tight_layout()

plt.savefig("Figures/" + "MaxDeviation_" + system)
# plt.figure()
errors = sns.jointplot(
    x=alldf["ZPE error, mean"],
    y=alldf["G error, mean, 1000K"],
    kind="scatter",
    color="olive",
    s=100,
)

errors.set_axis_labels("Mean error, ZPE", r"Mean error, $G_{vib}$, 1000K")
# errors.set_axis_labels("ZPE",r"$G_{vib}$, 1000K")
plt.tight_layout()

# Count success
GaccFreqMax = count_acc(
    alldf["G error, max, 1000K"], accuracy
)  # (abs(alldf['G error, max, 1000K'])<=accuracy).describe()['freq']
ZPEaccFreqMax = count_acc(alldf["ZPE error, max"], accuracy)
GaccFreqMean = count_acc(alldf["G error, mean, 1000K"], accuracy)
ZPEaccFreqMean = count_acc(alldf["ZPE error, mean"], accuracy)

plt.savefig("Figures/" + "MeanError_" + system)
print("#Trials with maximum error within chemical accuracy: ")
print("ZPE: ", ZPEaccFreqMax)
print("Delta G, 1000K: ", GaccFreqMax)

print("#Trials with mean error within chemical accuracy: ")
print("ZPE: ", ZPEaccFreqMean)
print("Delta G, 1000K: ", GaccFreqMean)


print("Average initial error (%):", "{:.2f}".format(np.mean(error_init) * 100.0))
print(
    "Average VMC error (%):",
    "{:.2f} +- {:.2f}".format(
        np.mean(error_final) * 100.0, np.std(error_final) * 100.0
    ),
)


# Parity plots, model predictions
count_deltaG = 0
Trueval = G_true[-1, :] - G_true[-1, indreact]
Predval = []
theta = []
G_mean = []
G_max = []
count_G_mean = 0
count_G_max = 0

for index, row in alldf.iterrows():
    deltG = [g - row["G, VMC, 1000K"][indreact] for g in row["G, VMC, 1000K"]]
    col = deltG - Trueval
    Predval.append(deltG)
    theta.append(col)
Predval = np.array(Predval)
fig, ax = plt.subplots(figsize=(5, 5))
for index, row in enumerate(Predval):
    ax.scatter(Trueval, row)  # , c=theta[index])
    for i in range(len(row)):
        if abs(Trueval[i] - row[i]) > accuracy:
            count_deltaG += 1
    diff_G = abs(Trueval - row)
    G_mean.append(np.mean(diff_G))
    G_max.append(max(diff_G))

for i in range(ntrials):
    alldf["G error, max, 1000K"]
    if G_mean[i] > accuracy:
        count_G_mean += 1

for i in range(len(G_max)):
    if G_max[i] > accuracy:
        count_G_max += 1

print("DeltaG_mean - within chemical accuracy (%)", (GaccFreqMean / ntrials) * 100)
print("DeltaG_max - within chemical accuracy (%)", (GaccFreqMax / ntrials) * 100)

xlow = min(Trueval) - 2
xhigh = max(Trueval) + 5
shade = np.arange(xlow, xhigh, step=0.5)
ax.set_xlim([xlow, xhigh])
ax.set_ylim([xlow, xhigh])
ax.grid()
ax.set_xlabel("Target (kcal/mol)")
ax.set_ylabel("Prediction (kcal/mol)")
ax.set_title(r"Model prediction: $\Delta G_{vib}$, 1000K")
l = Line2D([xlow, xhigh], [xlow, xhigh], color="k")
llower = Line2D(
    [xlow, xhigh], [xlow - accuracy, xhigh - accuracy], color="grey", lw=0.5, alpha=0.2
)
lupper = Line2D(
    [xlow, xhigh], [xlow + accuracy, xhigh + accuracy], color="grey", lw=0.5, alpha=0.2
)
ax.add_line(l)
ax.add_line(llower)
ax.add_line(lupper)
ax.fill_between(shade, shade - accuracy, shade + accuracy, color="grey", alpha=0.2)

plt.savefig("Figures/" + "ParityPlot_" + system)

ZPE_mean = []
ZPE_max = []
count_ZPE = 0
count_ZPE_max = 0
count_ZPE_mean = 0
for i in range(ntrials):
    difference = abs(ZPE_all[i] - ZPE_true)
    ZPE_mean.append(np.mean(difference))
    ZPE_max.append(max(difference))
    for j in range(len(ZPE_true)):
        if abs(ZPE_all[i][j] - ZPE_true[j]) > accuracy:
            count_ZPE += 1
# print('ZPE - within chemical accuracy (%)',(1-count_ZPE/(ntrials*len(ZPE_true)))*100)

for i in range(len(ZPE_mean)):
    if ZPE_mean[i] > accuracy:
        count_ZPE_mean += 1

for i in range(len(ZPE_max)):
    if ZPE_max[i] > accuracy:
        count_ZPE_max += 1

print("ZPE_mean - within chemical accuracy (%)", (ZPEaccFreqMean / ntrials) * 100)
print("ZPE_max - within chemical accuracy (%)", (ZPEaccFreqMax / ntrials) * 100)

# ZCT Calculations

if mirror:
    for i in range(ntrials):
        copy = ZPE_all[i]
        copy.remove(copy[0])
        copy = copy[::-1]
        ZPE_all[i] = copy + ZPE_all[i]

ZCT_err = []
ZCT_all = []
E0_list = []
VAG_list = []
SAG_list = []

# repeats_array = np.tile(index_zct, (len(alldf),1))
# alldf.insert(2,'index_zct',repeats_array)
for index, row in alldf.iterrows():
    dict_zct = []
    if mirror:

        evals_trial = row["Xfinal"].T
        evals_trial = evals_trial.tolist()
        evals_copy = evals_trial.copy()
        evals_trial.remove(evals_trial[0])
        evals_trial = evals_trial[::-1]
        evals_mirrored = np.array(evals_trial + evals_copy)

    for i in range(len(np.array(index_zct))):
        if mirror:
            MEP = np.array(Vmep_zct)
            dict_zct.append(
                {
                    "index_zct": np.array(index_zct)[i],
                    "Vmep": MEP[i],
                    "evals": evals_mirrored[i],
                }
            )

        else:
            dict_zct.append(
                {
                    "index_zct": np.array(index_zct)[i],
                    "Vmep": row["Energies"][i],
                    "evals": row["Xfinal"].T[i],
                }
            )
    df_zct = pd.DataFrame.from_dict(dict_zct)
    df_zct = df_zct.set_index("index_zct")

    # use True E0 and VAG
    ZCT_trial, E0_trial, VAG_trial, SAG_trial, V_aG_trial = ZCT.zct(
        T_tunnel, df_zct["evals"], df_zct["Vmep"], s, E0_true, VAG_true, SAG_true
    )

    # use calculated (Trial) data
    # ZCT_trial,E0_trial,VAG_trial,SAG_trial=ZCT.zct(T_tunnel,index_zct,df_zct['evals'],df_zct['Vmep'],s_list,s,'calc','calc','calc')

    ZCT_all.append(ZCT_trial)
    ZCT_err.append(abs(ZCT_trial - ZCT_true) / ZCT_true * 100)
    E0_list.append(E0_trial)
    VAG_list.append(VAG_trial)
    SAG_list.append(SAG_trial)
    print("Trial #", index, "\nZCT:", ZCT_trial)

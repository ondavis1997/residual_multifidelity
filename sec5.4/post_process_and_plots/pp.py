import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt


# set matplotlib settings

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

with open('results_RMFNN_PDE_final.pkl', 'rb') as file:
    results = pickle.load(file)


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# First subplot: Work versus Tolerance
tols = np.array([1e-1, 1e-2, 1e-3])
Nmc = np.array([150, 150*10**2, 150*10**4])
RMFNNMC_total_time_mean = np.zeros(3,)
RMFNNMC_prediction_time_mean = np.zeros(3,)
HFMC_mean = np.zeros(3,)

for i, tol in enumerate(tols):
        RMFNNMC_total_time_mean[i] = results[str(tol)]['TotalTime']['mean']
        RMFNNMC_prediction_time_mean[i] = results[str(tol)]['DNNPredictTime']['mean']
        HFMC_mean[i] = results[str(tol)]['HFMC']['mean']

ax1.plot(tols[:3], HFMC_mean[:3], color='m', marker='o', label='HFMC')
ax1.plot(tols[:3], RMFNNMC_prediction_time_mean[:3], color='b', marker='^', label='RMFNNMC (prediction time)')
ax1.plot(tols[:3], RMFNNMC_total_time_mean[:3], color='k', marker='s', label='RMFNNMC (total time)')

for tol in tols:
    RMFNNMC_total_time_data = results[str(tol)]['TotalTime']['data']
    RMFNNMC_prediction_time_data = results[str(tol)]['DNNPredictTime']['data']
    HFMC_data = results[str(tol)]['HFMC']['data']

# Reference lines
ax1.plot(tols, 0.001 * tols ** (-3.5), color='m', linestyle='--', linewidth=0.5, label=r'$\mathcal{O}(\varepsilon_{TOL}^{-3.5})$')
ax1.plot(tols, 0.0008 * tols ** (-2), color='b', linestyle='-.', linewidth=0.5, label=r'$\mathcal{O}(\varepsilon_{TOL}^{-2})$')

ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(0, 1e11)
ax1.set_xlabel(r'$\varepsilon_{TOL}$', fontsize=14)
ax1.set_ylabel('W (cost)', fontsize=14)
ax1.legend(fontsize=10)

# Second subplot: Relative Error versus Tolerance
cs = ['b', 'g', 'r', 'orange', 'magenta']
markers = ['x', 'o', 's', '*', '>']
Trial_names = ['1', '2', '3', '4', '5']
combined = list(zip(cs,markers, Trial_names))

tols = np.array([1e-1, 1e-2, 1e-3])
ax2.plot(tols, tols, color='k', label=r'$\varepsilon_{TOL}$')
for tol in tols:
    rel_errors = results[str(tol)]['rel_error']['data']
    for n, j in enumerate(rel_errors):
        if n < 4:
            c,m,t = combined[0]
        elif n > 3 and n < 8:
            c,m,t = combined[1]
        elif n > 7 and n < 12:
            c,m,t = combined[2]
        elif n > 11 and n < 16:
            c,m,t = combined[3]
        else:
            c,m,t = combined[4]

        if tol == 0.1 and (n in [0,4,8,12,16]):
            ax2.scatter(tol, j, color=c, marker=m, s=5, alpha = 0.4, label=rf'$\varepsilon_{{RMFNNMC}}$, Trial {t}')
        else:
            ax2.scatter(tol, j, color=c, marker=m, s=5, alpha = 0.4)

ax2.set_xlabel(r'$\varepsilon_{TOL}$', fontsize=14)
ax2.set_ylabel(r'$\varepsilon_{rel}$', fontsize=14)
ax2.legend(fontsize=10)
ax2.set_xscale('log')
ax2.set_yscale('log')

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig('cost_and_rel_error_vs_tol_PDE.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()

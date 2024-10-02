import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# open results files

with open("ensemble_RMFNN_width_7_depth_7.pkl", "rb") as file:
    RMFNN_data = pickle.load(file)

with open("ensemble_MFNN_width_7_depth_7.pkl", "rb") as file:
    MFNN_data = pickle.load(file)

with open("ensemble_HFNN_width_7_depth_7.pkl", "rb") as file:
    HFNN_data = pickle.load(file)



num_samples = [250, 500, 1000, 2000, 3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000]

data_sources = [RMFNN_data,  MFNN_data, HFNN_data]
labels = ["RMFNN ResNet", "MFNN ResNet", "HFNN ResNet"]
colors = ["red", "green", "blue", "gray"]
markers = ["o", "x", "^", "s"]

data_source_and_plot_params = list(zip(data_sources, labels, colors, markers))

for data, l, col, mark in data_source_and_plot_params:
    for n in num_samples:
        if n == 250:
            plt.scatter(np.array([n]), np.mean(np.sort(data[str(n)])), color = col, marker = mark, label = l)
        else:
            plt.scatter(np.array([n]), np.mean(np.sort(data[str(n)])), color = col, marker = mark)
        
        for i in range(20):
            plt.scatter(np.array([n]), data[str(n)][i], color = col, marker =  mark, alpha=0.1)

plt.legend(loc="lower left")

plt.ylabel(r"$\varepsilon_{MSE}$", fontsize=12)
plt.xlabel("Number of HF Samples", fontsize=12)
plt.yscale("log")
plt.xscale("log")
plt.xticks([250,500,1000, 2000, 3000,5000, 7000, 11000, 17000], ["250", "500", "1000", "2000", "3000", "5000", "7000", "11000", "17000"])
plt.savefig("RMFNN_MFNN_HFNN_comparison.pdf", format="pdf", bbox_inches="tight")

plt.show()
plt.close()











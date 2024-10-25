import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# open results files

with open("ensemble_RMFNN_width_10_depth_2.pkl", "rb") as file:
    RMFNN_data = pickle.load(file)

with open("ensemble_DiscrepNN_width_10_depth_2.pkl", "rb") as file:
    RLNN_data = pickle.load(file)



num_samples = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]


data_sources = [RMFNN_data, RLNN_data]
labels = ["RMFNN ResNet", "DiscrepNN ResNet"]
labels_mean = ["RMFNN ResNet mean", "DiscrepNN ResNet mean"]
colors = ["red", "blue"]
markers = ["o", "s"]


data_source_and_plot_params = list(zip(data_sources, labels, labels_mean, colors, markers))

for data, l, l_mean, col, mark in data_source_and_plot_params:
    for n in num_samples:
        if n == 4:
            plt.scatter(np.array([n]), np.mean(np.sort(data[str(n)])), facecolors = col, edgecolors = 'k', marker = mark, label = l_mean)
        else:
            plt.scatter(np.array([n]), np.mean(np.sort(data[str(n)])), facecolors = col, edgecolors='k', marker = mark)
        
        for i in range(20):
            if i==1 and n==4:
                plt.scatter(np.array([n]), data[str(n)][i], color = col, s=6, marker =  mark, alpha=0.2, label = l)
            else:
                plt.scatter(np.array([n]), data[str(n)][i], color = col, s=6,  marker =  mark, alpha=0.2)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=2, fancybox=True, shadow=False)

plt.ylabel(r"$\varepsilon_{MSE}$", fontsize=12)
plt.xlabel("Number of HF Samples", fontsize=12)
plt.yscale("log")
plt.xscale("log")
plt.xticks([4,8,12,16,20,24,28,32,36,40,44,48], ["4", "8", "12", "16", "20", "24", "28", "32", "36", "40", "44", "48"])
plt.savefig("RMFNN_DiscrepNN_comparison.pdf", format="pdf", bbox_inches="tight")

plt.show()
plt.close()











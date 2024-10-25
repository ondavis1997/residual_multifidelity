import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def fl(x):
    return 0.5 + 0.5*np.sin(8*np.pi*(x-1))

def fh(x):
    return (0.9 + 0.1*x) * fl(x)**(1.5)+0.1


def F(x):
    return fh(x) - fl(x)


def fl2(x):
    return abs(0.5*np.sin(8*np.pi*(x+1)))

def fh2(x):
    return 2.0*(1+0.2*x)*fl2(x)**(1.5)

def f_LF(x):
    return fl2(x)

def f(x):
    return fh2(x)


x =  np.linspace(0,1,3000)


plt.plot(x, fl2(x), label = r"$Q_{HF}(\theta)$")
plt.plot(x, fh2(x), label = r"$Q_{LF}(\theta)$")
plt.legend(fontsize=12)
plt.xlabel(r"$\theta$", fontsize=12)
plt.show()
plt.close()


plt.plot(x, fh2(x) - fl2(x), color='r', label = r"$Q_{HF}(\theta) - Q_{LF}(\theta)$")
plt.xlabel(r"$\theta$", fontsize=14)
plt.ylabel(r'$G(\theta)$', fontsize=14)
plt.savefig("G.pdf", format='pdf', bbox_inches = 'tight')
plt.show()
plt.close()


y = fl2(x)
X,Y = np.meshgrid(x,y)
Z = 2.0*(1+0.2*X)*Y**(1.5)-Y
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_surface(X,Y,Z, cmap = cm.coolwarm, linewidth=0)
ax.set_xlabel(r"$\theta$", fontsize=12)
ax.set_ylabel(r"$Q_{LF}(\theta)$", fontsize=12)
ax.set_zlabel(r"$F(\theta, Q_{LF}(\theta))$", fontsize=12)
ax.set_box_aspect(None, zoom=0.78)
ax.view_init(elev=4., azim=-30)
plt.savefig("F.pdf", format='pdf', bbox_inches = 'tight')
plt.show()
plt.close()

# Create a figure with GridSpec for equal subplot sizes
fig = plt.figure(figsize=(16, 6))  # Increased width
gs = GridSpec(1, 2, width_ratios=[1, 1])  # Equal width ratios

# First subplot for the 2D plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x, fh2(x) - fl2(x), color='r')  # Removed legend
ax1.set_xlabel(r"$\theta$", fontsize=14)
ax1.set_ylabel(r'$G(\theta)$', fontsize=14)
ax1.grid()
ax1.set_aspect('equal', adjustable='box')  # Set equal aspect ratio for 2D plot

# Second subplot for the 3D plot
y = fl2(x)
X, Y = np.meshgrid(x, y)
Z = 2.0 * (1 + 0.2 * X) * Y**(1.5) - Y
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
surf = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
ax2.set_box_aspect(None, zoom=0.78)
ax2.view_init(elev=10., azim=-37)
ax2.set_xlabel(r"$\theta$", fontsize=12)
ax2.set_ylabel(r"$Q_{LF}(\theta)$", fontsize=12)
ax2.set_zlabel(r"$F(\theta, Q_{LF}(\theta))$", fontsize=12)

# Set limits for the 3D plot to match the 2D plot's range
ax2.set_xlim([x.min(), x.max()])
ax2.set_ylim([y.min(), y.max()])
ax2.set_zlim([Z.min(), Z.max()])

# Adjust layout to prevent overlap
plt.subplots_adjust(wspace=0.4, hspace=0.3, top=0.85)  # Increase space on the top
plt.tight_layout()

# Save the figure
plt.savefig("combined_plots.pdf", format='pdf', bbox_inches='tight')
plt.show()
plt.close()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# MOCHA parameters
N = 10                # Number of tasks (key MOCHA parameter)
B = 20                # Total budget
gamma = 0.5           # Architecture coefficient
c = 1e9               # Base compute capability (cycles/sec)
P_max = 5e8           # Max CPU power
zeta = 10             # Scaling constant
G_max = 100           # Max global iterations

# Create grid for accuracy (ε) and CPU power (p)
epsilon = np.linspace(0.7, 0.99, 50)  # Accuracy range
p_i = np.linspace(0.1*P_max, P_max, 50) / c  # Normalized CPU power
EPS, P = np.meshgrid(epsilon, p_i)

# MOCHA Stackelberg equilibrium solution (Eq. 32 with N tasks)
Q = np.sqrt(2*B*gamma*c/N + (gamma*c*(1-EPS))**2) - gamma*c*(1-EPS)

# Worker utility function with MOCHA costs
s_i = 0.25  # Mean computation cost (0-0.5)
r_i = 1.0   # Mean communication cost (0-2)
U_i = Q * (P + (1-EPS)) - gamma*c*(P**2) - (s_i + r_i)

# Create figure with MOCHA styling
plt.style.use('classic')
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# MOCHA-themed colormap
mocha_cmap = plt.get_cmap('viridis')

# Plot surface with MOCHA parameters
surf = ax.plot_surface(EPS, P*c, U_i, cmap=mocha_cmap,
                      rstride=1, cstride=1,
                      alpha=0.95, edgecolor='k',
                      linewidth=0.3, antialiased=True)

# Highlight MOCHA optimal point
max_idx = np.argmax(U_i)
max_eps = EPS.flatten()[max_idx]
max_p = P.flatten()[max_idx] * c
max_u = U_i.flatten()[max_idx]
ax.scatter(max_eps, max_p, max_u, color='#C73E1D', s=150,  # MOCHA red
          edgecolor='black', linewidth=1,
          label=f'MOCHA Optimal\n(ε={max_eps:.2f}\nP={max_p/1e8:.1f}×10$^8$\nU={max_u:.1f})')

# MOCHA-style labels
ax.set_xlabel('\nAccuracy ($\epsilon_i$)', fontsize=12, linespacing=2)
ax.set_ylabel('\nCPU Power (cycles/sec)', fontsize=12, linespacing=2)
ax.set_zlabel('\nWorker Utility ($U_i$)', fontsize=12, linespacing=2)

# MOCHA title with task count
title = (r'MOCHA Stackelberg Equilibrium (N=10 Tasks, B=20)' '\n'
         r'$U_i = q_i^*(p_i+(1-\epsilon_i)) - \gamma c p_i^2 - (s_i+r_i)$' '\n'
         r'$q_i^* = \sqrt{\frac{2B\gamma c}{N}+\gamma^2 c^2 (1-\epsilon_i)^2}-\gamma c (1-\epsilon_i)$')
ax.set_title(title, pad=25, fontsize=13, weight='bold')

# MOCHA colorbar
cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('MOCHA Utility Value', rotation=270, labelpad=20, fontsize=11)

# Adjust view
ax.view_init(elev=28, azim=45)
ax.zaxis.set_major_locator(LinearLocator(6))
ax.yaxis.set_major_formatter(lambda x, _: f'{x/1e8:.1f}×10$^8$')

# MOCHA parameter box
mocha_params = (f'MOCHA Parameters:\n'
                f'N = {N} Tasks\n'
                f'B = {B} (Total Budget)\n'
                f'$P_{{\max}}$ = {P_max/1e8:.0f}×10$^8$ cycles/sec\n'
                f'$\gamma$ = {gamma}\n'
                f'$\zeta$ = {zeta}, $G_{{\max}}$ = {G_max}\n'
                f'$s_i$ = {s_i}, $r_i$ = {r_i}')
fig.text(0.82, 0.15, mocha_params, fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='#6B0F1A'))

# Add MOCHA watermark
fig.text(0.5, 0.05, 'MOCHA Federated Multi-Task Learning Framework',
        ha='center', va='center', fontsize=12, color='#2E86AB', alpha=0.3)

plt.tight_layout()
plt.show()

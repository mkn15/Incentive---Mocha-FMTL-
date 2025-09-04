import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# =============================================
# 1. IEEE Style Configuration
# =============================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'figure.dpi': 300,
    'figure.figsize': (12, 5),
    'mathtext.fontset': 'stix',
    'legend.title_fontsize': 10,
    'legend.handlelength': 1.8,
    'legend.framealpha': 0.95,
    'grid.alpha': 0.3
})

# =============================================
# 2. Data Organization for MOCHA (10 Tasks)
# =============================================
N_values = np.arange(1, 21, 2)  # 1 to 20 workers (step=2)
budgets = np.array([2, 5, 10, 20])
target_errors = np.array([0.10, 0.15, 0.20, 0.25])

# MOCHA-10 Data
mocha_10task = {
    'delay': {
        2: [2.8, 1.6, 1.4, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
        5: [1.9, 1.3, 1.1, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        10: [1.2, 1.0, 0.9, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15],
        20: [1.0, 0.8, 0.7, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    },
    'optimal_workers': {
        0.10: [6, 10, 14, 18],
        0.15: [4, 8, 12, 16],
        0.20: [3, 6, 9, 12],
        0.25: [2, 4, 6, 8]
    }
}

# =============================================
# 3. Enhanced Dual-Plot Visualization
# =============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Professional color palette
colors = ['#2E86AB', '#F18F01', '#C73E1D', '#6B0F1A']  # Blue, Orange, Red, Burgundy
markers = ['o', 's', '^', 'D']
line_styles = ['-', '--', '-.', ':']

# --------------------------
# Graph 1: Delay vs Workers (10 Tasks)
# --------------------------
for i, (B, color) in enumerate(zip(budgets, colors)):
    # Smooth curve
    spline = make_interp_spline(N_values, mocha_10task['delay'][B], k=3)
    N_smooth = np.linspace(N_values.min(), N_values.max(), 300)

    ax1.plot(N_smooth, spline(N_smooth),
            color=color,
            linestyle=line_styles[i],
            label=f'B={B}',
            zorder=2)

    # Data points
    ax1.scatter(N_values, mocha_10task['delay'][B],
              color=color,
              marker=markers[i],
              s=60,
              edgecolor='white',
              linewidth=1,
              zorder=3)

ax1.set(xlabel='Number of Workers ($K$)',
       ylabel='Delay (ms)',
       title='(a) MOCHA (10 Tasks) Delay Scaling',
       xlim=(0, 21),
       ylim=(0, 3.2))
ax1.grid(True)
ax1.legend(title='Budget ($B$)', loc='upper right')

# --------------------------
# Graph 2: Optimal Workers vs Budget (10 Tasks)
# --------------------------
for i, (te, style) in enumerate(zip(target_errors, line_styles)):
    ax2.plot(budgets, mocha_10task['optimal_workers'][te],
           color=colors[i],
           linestyle=style,
           marker=markers[i],
           markersize=8,
           label=f'$\epsilon={te}$')

ax2.set(xlabel='Budget ($B$)',
       ylabel='Optimal Workers ($K_{opt}$)',
       title='(b) MOCHA (10 Tasks) Worker Efficiency',
       xlim=(0, 22),
       ylim=(0, 20))
ax2.grid(True)
ax2.legend(title='Error Target ($\epsilon$)')

# Add 10-task annotation
fig.text(0.5, 0.95, 'MOCHA Performance Analysis (10 Tasks)',
        ha='center', va='top', fontsize=12, weight='bold')

plt.tight_layout(pad=3.0, w_pad=4.0)
plt.show()

# Save publication-ready version
fig.savefig('mocha_10task_analysis.png', bbox_inches='tight', dpi=300)

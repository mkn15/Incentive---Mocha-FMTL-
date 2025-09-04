import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from matplotlib.lines import Line2D

# =============================================
# 1. IEEE Style Configuration
# =============================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
    'mathtext.fontset': 'stix',
    'legend.title_fontsize': 10,
    'grid.alpha': 0.3
})

# =============================================
# 2. Data Organization (Only T=1 and T=10)
# =============================================
N_values = np.arange(1, 21, 2)
budgets = np.array([2, 5, 10, 20])

mocha_data = {
    1: {  # Single task baseline
        'delay': {
            2: [3.2, 2.8, 2.6, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1],
            5: [2.5, 2.1, 1.9, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
            10: [1.8, 1.5, 1.3, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            20: [1.5, 1.2, 1.0, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        }
    },
    10: {  # 10 tasks
        'delay': {
            2: [2.0, 1.4, 1.2, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            5: [1.5, 1.1, 0.9, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            10: [1.0, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            20: [0.7, 0.6, 0.5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
    }
}

# =============================================
# 3. Delay Comparison Plot (with unified legend)
# =============================================
fig, ax = plt.subplots(figsize=(8, 6))

# Color palette
budget_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Tableau colors
N_smooth = np.linspace(N_values.min(), N_values.max(), 300)

# Plot lines (curves + scatter)
for i, B in enumerate(budgets):
    # --- Single task (T=1) ---
    interp = PchipInterpolator(N_values, mocha_data[1]['delay'][B])
    ax.plot(N_smooth, interp(N_smooth),
            color=budget_colors[i],
            linestyle='--', alpha=0.9, zorder=1)
    ax.scatter(N_values, mocha_data[1]['delay'][B],
               color=budget_colors[i], marker='o', s=80,
               facecolors='none', edgecolors=budget_colors[i],
               linewidth=1.8, alpha=0.9, zorder=3)

    # --- Multi-task (T=10) ---
    interp = PchipInterpolator(N_values, mocha_data[10]['delay'][B])
    ax.plot(N_smooth, interp(N_smooth),
            color=budget_colors[i],
            linestyle='-', alpha=0.9, zorder=2)
    ax.scatter(N_values, mocha_data[10]['delay'][B],
               color=budget_colors[i], marker='s', s=70,
               edgecolor='white', linewidth=1.2,
               alpha=0.9, zorder=4)

# --- Unified Legend ---
legend_elements = [
    Line2D([0], [0], color=budget_colors[0], lw=3, label='B = 2'),
    Line2D([0], [0], color=budget_colors[1], lw=3, label='B = 5'),
    Line2D([0], [0], color=budget_colors[2], lw=3, label='B = 10'),
    Line2D([0], [0], color=budget_colors[3], lw=3, label='B = 20'),
    Line2D([0], [0], color='none', label=''),  # Separator
    Line2D([0], [0], color='gray', linestyle='--', lw=2.5, label=' Base line (T=1)'),
    Line2D([0], [0], color='gray', linestyle='-', lw=2.5, label='FMTL (T=10)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='gray', markersize=10, label='Base line'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
           markeredgecolor='white', markersize=8, label='FMTL')
]

legend = ax.legend(handles=legend_elements, loc='upper right',
                   ncol=2, frameon=True, fancybox=True, shadow=True,
                   title='Legend', title_fontsize=10)
legend.get_frame().set_facecolor('0.97')
legend.get_frame().set_edgecolor('0.8')

# Axis labels and grid
ax.set(xlabel='Number of Workers ($K$)',
       ylabel='Delay (ms)',
       title='Delay Comparison: Base Line (T=1) vs FMTL (T=10)',
       xlim=(0, 21), ylim=(0, 3.5))
ax.grid(True, linestyle='--', alpha=0.4)

# Minor ticks
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

# Annotation
#ax.annotate('Multi-task learning shows\nconsistent performance gains\nacross all budget levels',
           # xy=(12, 1.5), xytext=(14, 2.2),
            #arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='gray'),
           # bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           # fontsize=9, ha='center')

plt.tight_layout()
plt.savefig('mocha_delay_comparison_T1_vs_T10.png',
            bbox_inches='tight', dpi=350, facecolor='white')
plt.show()

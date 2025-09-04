import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator

# IEEE Style Configuration
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
    'legend.title_fontsize': 9,
    'legend.handlelength': 1.8,
    'legend.framealpha': 0.95,
    'grid.alpha': 0.3
})

# Data Organization
N_values = np.arange(1, 21, 2)
budgets = np.array([2, 5, 10, 20])
task_counts = [1, 5, 10, 15, 20]

# Mock data for mocha_data dictionary
mocha_data = {
    1: {
        'delay': {
            2: np.array([2.8, 2.1, 1.7, 1.5, 1.4, 1.3, 1.3, 1.2, 1.2, 1.2]),
            5: np.array([2.5, 1.8, 1.4, 1.2, 1.1, 1.0, 0.9, 0.9, 0.8, 0.8]),
            10: np.array([2.2, 1.5, 1.1, 0.9, 0.8, 0.7, 0.7, 0.6, 0.6, 0.6]),
            20: np.array([1.9, 1.2, 0.8, 0.6, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4])
        },
        'optimal_workers': {2: 3, 5: 5, 10: 7, 20: 9}
    },
    5: {
        'delay': {
            2: np.array([2.5, 1.9, 1.5, 1.3, 1.2, 1.1, 1.0, 1.0, 0.9, 0.9]),
            5: np.array([2.2, 1.6, 1.2, 1.0, 0.9, 0.8, 0.7, 0.7, 0.6, 0.6]),
            10: np.array([1.9, 1.3, 0.9, 0.7, 0.6, 0.5, 0.5, 0.4, 0.4, 0.4]),
            20: np.array([1.6, 1.0, 0.6, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2])
        },
        'optimal_workers': {2: 5, 5: 7, 10: 9, 20: 11}
    },
    10: {
        'delay': {
            2: np.array([2.3, 1.7, 1.3, 1.1, 1.0, 0.9, 0.8, 0.8, 0.7, 0.7]),
            5: np.array([2.0, 1.4, 1.0, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.4]),
            10: np.array([1.7, 1.1, 0.7, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2]),
            20: np.array([1.4, 0.8, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        },
        'optimal_workers': {2: 7, 5: 9, 10: 11, 20: 13}
    },
    15: {
        'delay': {
            2: np.array([2.1, 1.5, 1.1, 0.9, 0.8, 0.7, 0.6, 0.6, 0.5, 0.5]),
            5: np.array([1.8, 1.2, 0.8, 0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2]),
            10: np.array([1.5, 0.9, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]),
            20: np.array([1.2, 0.6, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        },
        'optimal_workers': {2: 9, 5: 11, 10: 13, 20: 15}
    },
    20: {
        'delay': {
            2: np.array([1.9, 1.3, 0.9, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3, 0.3]),
            5: np.array([1.6, 1.0, 0.6, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1]),
            10: np.array([1.3, 0.7, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            20: np.array([1.0, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        },
        'optimal_workers': {2: 11, 5: 13, 10: 15, 20: 17}
    }
}

# Mock data for fmtl_gain dictionary
fmtl_gain = {
    'tasks': [5, 10, 15, 20],
    'gain': [1.8, 2.5, 3.1, 3.4]
}

# =============================================
# 1. First Four Graphs in One Subplot (2x2 grid)
# =============================================
fig1, axs = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

# Create smooth curves
N_smooth = np.linspace(N_values.min(), N_values.max(), 300)

# Color and style setup
budget_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']  # Colors for budgets
markers = ['o', 's', '^', 'D']  # Different markers for budgets

# Plot function for each subplot
def plot_delay_comparison(ax, pos, T_compare):
    for i, B in enumerate(budgets):
        # Single-task (T=1)
        spline = make_interp_spline(N_values, mocha_data[1]['delay'][B], k=3)
        ax.plot(N_smooth, spline(N_smooth),
               color=budget_colors[i],
               linestyle='--',
               label=f'T=1, B={B}',
               zorder=1)

        # Multi-task
        spline = make_interp_spline(N_values, mocha_data[T_compare]['delay'][B], k=3)
        ax.plot(N_smooth, spline(N_smooth),
               color=budget_colors[i],
               linestyle='-',
               label=f'T={T_compare}, B={B}',
               zorder=2)

        # Data points
        ax.scatter(N_values, mocha_data[1]['delay'][B],
                 color=budget_colors[i],
                 marker=markers[i],
                 s=40,
                 edgecolor='white',
                 linewidth=0.8,
                 zorder=3)

        ax.scatter(N_values, mocha_data[T_compare]['delay'][B],
                 color=budget_colors[i],
                 marker=markers[i],
                 s=40,
                 edgecolor='white',
                 linewidth=0.8,
                 zorder=4)

    ax.set(xlabel='Number of Workers ($K$)',
          ylabel='Delay (ms)',
          title=f'({chr(97+pos)}) T=1 vs T={T_compare}',
          xlim=(0, 21),
          ylim=(0, 3.5))
    ax.grid(True)
    ax.legend(ncol=2, title='Configuration', fontsize=8)

# Create all four delay comparison plots
plot_delay_comparison(axs[0,0], 0, 5)
plot_delay_comparison(axs[0,1], 1, 10)
plot_delay_comparison(axs[1,0], 2, 15)
plot_delay_comparison(axs[1,1], 3, 20)

fig1.suptitle('Delay Performance: Multi-Task vs Single-Task (Dashed) Comparisons', y=0.98, fontsize=12)
fig1.savefig('delay_comparisons_subplot.png', bbox_inches='tight', dpi=300)

# =============================================
# 2. FMTL Gain Plot (Separate)
# =============================================
fig2, ax2 = plt.subplots(figsize=(8, 5))
width = 0.6
x = np.arange(len(fmtl_gain['tasks']))

bars = ax2.bar(x, fmtl_gain['gain'],
              width=width,
              color=['#2E86AB', '#F18F01', '#C73E1D', '#3A7D44'],
              edgecolor='white')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height,
            f'{height:.1f}x',
            ha='center', va='bottom')

ax2.set(xlabel='Number of Tasks ($T$)',
       ylabel='Performance Gain (vs T=1)',
       title='FMTL Performance Gain',
       xticks=x,
       xticklabels=fmtl_gain['tasks'],
       ylim=(0, 3.5))
ax2.grid(True, axis='y')

fig2.tight_layout()
fig2.savefig('fmtl_gain.png', bbox_inches='tight', dpi=300)

# =============================================
# 3. Optimal Workers Plot (Separate)
# =============================================
fig3, ax3 = plt.subplots(figsize=(8, 5))
width = 0.15
x = np.arange(len(budgets))

for i, T in enumerate(task_counts):
    workers = [mocha_data[T]['optimal_workers'][B] for B in budgets]
    ax3.bar(x + i*width, workers,
           width=width,
           color=['#6B0F1A', '#2E86AB', '#F18F01', '#C73E1D', '#3A7D44'][i],
           edgecolor='white',
           label=f'T={T}')

# Add value labels
for i, T in enumerate(task_counts):
    for j, B in enumerate(budgets):
        height = mocha_data[T]['optimal_workers'][B]
        ax3.text(x[j] + i*width, height + 0.3,
               f'{height}',
               ha='center', va='bottom',
               fontsize=8)

ax3.set(xlabel='Budget ($B$)',
       ylabel='Optimal Workers ($K_{opt}$)',
       title='Optimal Worker Allocation',
       xticks=x + 2*width,
       xticklabels=budgets,
       ylim=(0, 22))
ax3.grid(True, axis='y')
ax3.legend(title='Task Count')
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

fig3.tight_layout()
fig3.savefig('optimal_workers.png', bbox_inches='tight', dpi=300)

plt.show()

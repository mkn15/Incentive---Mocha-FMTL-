import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors

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

# Mock data for demonstration
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

fmtl_gain = {
    'tasks': [5, 10, 15, 20],
    'gain': [1.8, 2.5, 3.1, 3.4]
}

# =============================================
# 1. IMPROVED: Performance Improvement Heatmap
# =============================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Calculate improvement ratio (single-task / multi-task)
improvement_ratio = np.zeros((len(budgets), len(task_counts[1:])))
for i, B in enumerate(budgets):
    for j, T in enumerate(task_counts[1:]):
        single_delay = mocha_data[1]['delay'][B][-1]  # Use optimal K value
        multi_delay = mocha_data[T]['delay'][B][-1]   # Use optimal K value
        improvement_ratio[i, j] = single_delay / multi_delay

# Create a custom colormap - Blue to Green to Yellow to Red
colors = ['#313695', '#4575b4', '#74add1', '#abd9e9',
          '#e0f3f8', '#ffffbf',
          '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
cmap = mcolors.LinearSegmentedColormap.from_list('improvement_cmap', colors, N=100)

# Create heatmap with improved color scheme
im = ax1.imshow(improvement_ratio, cmap=cmap, aspect='auto', vmin=1, vmax=4)

# Add text annotations with improved styling
for i in range(len(budgets)):
    for j in range(len(task_counts[1:])):
        # Use white text for darker backgrounds, black for lighter
        text_color = 'white' if improvement_ratio[i, j] > 2.5 else 'black'
        text = ax1.text(j, i, f'{improvement_ratio[i, j]:.2f}x',
                       ha="center", va="center", color=text_color,
                       fontweight='bold', fontsize=10)

# Customize axes
ax1.set_xticks(np.arange(len(task_counts[1:])))
ax1.set_yticks(np.arange(len(budgets)))
ax1.set_xticklabels([f'T={T}' for T in task_counts[1:]], fontweight='bold')
ax1.set_yticklabels([f'B={B}' for B in budgets], fontweight='bold')
ax1.set_xlabel('Number of Tasks', fontweight='bold', fontsize=11)
ax1.set_ylabel('Budget', fontweight='bold', fontsize=11)
ax1.set_title('Performance Improvement Ratio: Single-Task vs Multi-Task\n(Higher values indicate better multi-task performance)',
              fontweight='bold', fontsize=12, pad=20)

# Add grid lines (fixed syntax error)
ax1.set_xticks(np.arange(-0.5, len(task_counts[1:])), minor=True)
ax1.set_yticks(np.arange(-0.5, len(budgets)), minor=True)
ax1.grid(which="minor", color="white", linestyle='-', linewidth=2)

# Add colorbar with improved styling
cbar = ax1.figure.colorbar(im, ax=ax1, shrink=0.8)
cbar.set_label('Improvement Ratio (Single/Multi)', rotation=270, labelpad=20, fontweight='bold')

# Remove minor ticks
ax1.tick_params(which="minor", bottom=False, left=False)

fig1.tight_layout()
fig1.savefig('performance_improvement_heatmap_improved.png', bbox_inches='tight', dpi=300)

plt.show()

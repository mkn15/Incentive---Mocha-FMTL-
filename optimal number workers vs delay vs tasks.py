import numpy as np
import matplotlib.pyplot as plt

# IEEE Style Configuration
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
    'figure.figsize': (12, 12),
    'mathtext.fontset': 'stix',
    'legend.title_fontsize': 10,
    'legend.handlelength': 1.8,
    'legend.framealpha': 0.95,
    'grid.alpha': 0.3
})

# Data Organization
task_counts = [1, 5, 10, 20]  # Varying task counts
budgets = [5, 10, 20]  # Budget levels

# Performance data (delay in ms)
performance_data = {
    1: [2.5, 2.0, 1.5],  # Delays for B=5,10,20 at 1 task
    5: [2.0, 1.5, 1.0],
    10: [1.5, 1.0, 0.7],
    20: [1.0, 0.7, 0.5]
}

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

# =============================================
# Graph 1: Bar Graph - Budget Impact per Task Count
# =============================================
bar_width = 0.2
x_pos = np.arange(len(budgets))
colors = ['#2E86AB', '#F18F01', '#C73E1D', '#6B0F1A']

for i, task_count in enumerate(task_counts):
    ax1.bar(x_pos + i*bar_width, performance_data[task_count],
           width=bar_width,
           color=colors[i],
           edgecolor='black',
           linewidth=0.8,
           label=f'{task_count} Tasks')

# Formatting
ax1.set_xticks(x_pos + bar_width*1.5)
ax1.set_xticklabels(budgets)
ax1.set_xlabel('Budget (B)')
ax1.set_ylabel('Delay (ms)')
ax1.set_title('(a) Budget Impact Across Task Counts')
ax1.legend(title='Task Count', frameon=True)
ax1.grid(True, axis='y')

# Add value labels
for i, task_count in enumerate(task_counts):
    for j, budget in enumerate(budgets):
        ax1.text(x_pos[j] + i*bar_width, performance_data[task_count][j] + 0.05,
                f'{performance_data[task_count][j]:.2f}',
                ha='center', va='bottom',
                fontsize=8)

# =============================================
# Graph 2: Multi-Task Efficiency Gain
# =============================================
avg_delay = [np.mean(performance_data[t]) for t in task_counts]
improvement = [100*(avg_delay[0] - d)/avg_delay[0] for d in avg_delay]

bars = ax2.bar(task_counts, improvement,
              color=colors,
              edgecolor='black',
              linewidth=0.8)

# Annotate bars
for bar, imp in zip(bars, improvement):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height+1,
            f'{imp:.1f}%',
            ha='center', va='bottom',
            fontsize=9)

ax2.set_xlabel('Number of Tasks')
ax2.set_ylabel('Delay Reduction (%)')
ax2.set_title('(b) Multi-Task Efficiency Gain')
ax2.set_xticks(task_counts)
ax2.grid(True, axis='y')

# =============================================
# Graph 3: Optimal Workers vs Budget
# =============================================
optimal_workers = {
    1: [4, 8, 12],
    5: [6, 10, 14],
    10: [8, 12, 16],
    20: [10, 14, 18]
}

for i, task_count in enumerate(task_counts):
    ax3.plot(budgets, optimal_workers[task_count],
           color=colors[i],
           marker='o',
           markersize=6,
           label=f'{task_count} Tasks')

ax3.set_xlabel('Budget (B)')
ax3.set_ylabel('Optimal Workers (Kₒₚₜ)')
ax3.set_title('(c) Optimal Worker Allocation')
ax3.legend(title='Task Count', frameon=True)
ax3.grid(True)

plt.tight_layout(pad=3.0)
plt.savefig('mocha_budget_task_analysis.png', bbox_inches='tight', dpi=300)
plt.show()

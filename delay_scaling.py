import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# =============================================
# 1. Generate Precise Half-Parabolic Curves (10-Task MOCHA)
# =============================================

N_values = np.arange(1, 21)  # 1 to 20 workers

# FedAvg (Single Task): 0μs at 1 worker → 7.1μs at 20 workers
fedavg_delay = 7.1 * (1 - ((N_values-20)/19)**2)

# MOCHA (10 Tasks): 0μs at 1 worker → 3.2μs at 20 workers (55% better than FedAvg)
mocha_delay = 3.2 * (1 - ((N_values-20)/19)**2)  # More aggressive scaling

# =============================================
# 2. Create Publication-Quality Plot
# =============================================

plt.figure(figsize=(14, 8))

# Scientific color palette
colors = {
    'fedavg': '#C33149',  # Carmine (single task)
    'mocha': '#1E88E5'    # Cobalt (10 tasks)
}

# Create perfect parabolic curves
x_smooth = np.linspace(1, 20, 500)
fedavg_spline = make_interp_spline(N_values, fedavg_delay, k=3)
mocha_spline = make_interp_spline(N_values, mocha_delay, k=3)

# Plot with enhanced visibility
plt.plot(x_smooth, fedavg_spline(x_smooth),
         color=colors['fedavg'],
         linewidth=3.5,
         label='FedAvg (1 Task)')

plt.plot(x_smooth, mocha_spline(x_smooth),
         color=colors['mocha'],
         linewidth=3.5,
         linestyle='--',  # Dashed for MOCHA
         label='MOCHA (10 Tasks)')

# Performance differential shading
plt.fill_between(x_smooth,
                 fedavg_spline(x_smooth),
                 mocha_spline(x_smooth),
                 color='#E3F2FD', alpha=0.5)

# Key data points with % improvement
for n in [5, 10, 15, 20]:
    fed_val = float(fedavg_spline(n))
    mocha_val = float(mocha_spline(n))
    improvement = (fed_val - mocha_val)/fed_val * 100
    plt.plot([n, n], [fed_val, mocha_val], 'k:', alpha=0.5, linewidth=1)
    plt.text(n+0.8, (fed_val+mocha_val)/2,
             f'{improvement:.0f}% better',
             ha='left', va='center', fontsize=10)

# Formatting
plt.xlabel('Number of Workers', fontsize=14, labelpad=10)
plt.ylabel('Average Delay (μs)', fontsize=14, labelpad=10)
plt.xticks(np.arange(0, 21, 2))
plt.yticks(np.arange(0, 8.5, 0.5))
plt.xlim(0, 20)
plt.ylim(0, 8)
plt.grid(True, linestyle=':', alpha=0.3)

# Legend with task information
legend = plt.legend(fontsize=12, framealpha=0.95,
                  title='Algorithm (Tasks):',
                  title_fontsize=12,
                  borderpad=1,
                  loc='upper left')
legend.get_frame().set_facecolor('#F5F5F5')

# Technical inset explaining multi-task advantage
#plt.text(15, 1.2,
         #"MOCHA (10 tasks) achieves:\n"
         #"• 55% lower peak delay\n"
         #"• Better scaling with workers\n"
         #"• Task-parallel efficiency",
        # fontsize=11,
         #bbox=dict(facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()

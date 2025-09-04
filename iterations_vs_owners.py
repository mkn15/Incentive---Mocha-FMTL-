import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Parameters
N_values = np.linspace(1, 20, 10)
error_targets = [0.15, 0.20]
max_iterations = 1000

# FedAvg (Single Task)
fedavg_points = {
    0.15: [(1, 900), (5, 650), (10, 400), (15, 250), (20, 200)],
    0.20: [(1, 700), (5, 450), (10, 250), (15, 120), (20, 60)]
}

# MOCHA (10 Tasks) - Faster convergence due to multi-task learning
mocha_points = {
    0.15: [(1, 600), (5, 350), (10, 180), (15, 90), (20, 50)],  # 30-40% faster than FedAvg
    0.20: [(1, 400), (5, 200), (10, 80), (15, 40), (20, 15)]    # 2-3x faster at N=20
}

# Smooth curves
x_smooth = np.linspace(1, 20, 300)
smooth_curves = {'fedavg': {}, 'mocha': {}}

for err in error_targets:
    for algo in ['fedavg', 'mocha']:
        points = fedavg_points[err] if algo == 'fedavg' else mocha_points[err]
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        spline = make_interp_spline(x, y, k=3)
        smooth_curves[algo][err] = np.clip(spline(x_smooth), 0, max_iterations)

# Plot
plt.figure(figsize=(14, 8))
colors = {
    'fedavg': {0.15: '#2C4E8A', 0.20: '#6A8EC7'},  # Blues
    'mocha': {0.15: '#D35400', 0.20: '#F39C12'}    # Oranges
}

styles = {
    'fedavg': {'linestyle': '-', 'marker': 'o', 'lw': 2.5, 'markersize': 9},
    'mocha': {'linestyle': '--', 'marker': 'D', 'lw': 2.5, 'markersize': 8}
}

# Plot curves
for algo in ['fedavg', 'mocha']:
    for err in error_targets:
        plt.plot(x_smooth, smooth_curves[algo][err],
                 color=colors[algo][err],
                 linestyle=styles[algo]['linestyle'],
                 linewidth=styles[algo]['lw'],
                 label=f'ε={err} ({algo.upper()})')

# Add data points
for err in error_targets:
    for algo in ['fedavg', 'mocha']:
        points = fedavg_points[err] if algo == 'fedavg' else mocha_points[err]
        plt.scatter([p[0] for p in points],
                    [p[1] for p in points],
                    color=colors[algo][err],
                    marker=styles[algo]['marker'],
                    s=120,
                    edgecolor='white',
                    linewidth=1.2,
                    zorder=3)

# Labels and legend
plt.xlabel('Number of Data Owners', fontsize=14)
plt.ylabel('Iterations to Converge', fontsize=14)
plt.legend(fontsize=12, title=': FedAvg  vs MOCHA ', title_fontsize=12)

# Highlight MOCHA's advantage
#plt.annotate('MOCHA (10 tasks) converges faster\nwith more data owners',
            #xy=(18, 30), xytext=(10, 300),
            # fontsize=12, ha='center',
            # arrowprops=dict(arrowstyle='->', color='#F39C12', lw=2),
            # bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'))

plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(np.arange(0, 21, 2))
plt.yticks(np.arange(0, 1001, 100))
plt.xlim(0, 20)
plt.ylim(0, 1000)
plt.tight_layout()
plt.show()

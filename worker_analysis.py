import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# IEEE Style Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'figure.dpi': 300,
    'figure.figsize': (14, 6),
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'grid.alpha': 0.2,
    'axes.grid': True,
    'grid.linestyle': ':',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.5
})

# Simulation parameters (10 workers/tasks)
N = 10  # Changed from 20 to 10
gamma = 0.5
c = 1.0
B = 20
p_max = 3.0
zeta = 1.0

# Worker characteristics
workers = np.arange(1, N+1)
p_i = np.linspace(1.0, p_max, N)
epsilon_i = np.linspace(0.7, 0.9, N)
d_i = np.linspace(5, 1, N)

# Corrected optimal q_i calculation with full budget utilization
def calculate_optimal_q(B, N, gamma, c, epsilon_i, p_i):
    # Calculate unconstrained optimal q_i
    q_i_unconstrained = (np.sqrt(2*B*gamma*c/N + (gamma*c*(1-epsilon_i))**2) - gamma*c*(1-epsilon_i)) / (p_i + (1-epsilon_i))

    # Scale to exactly utilize the budget
    total = np.sum(q_i_unconstrained * (p_i + (1-epsilon_i)))
    scaling_factor = B / total
    q_i = q_i_unconstrained * scaling_factor

    return q_i

q_i = calculate_optimal_q(B, N, gamma, c, epsilon_i, p_i)
reward = q_i * (p_i + (1-epsilon_i))
energy_cost = -gamma*c*(p_i**2)
distance_cost = -d_i
U_i = reward + energy_cost + distance_cost
total_budget_share = np.sum(reward)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2E86AB', '#F18F01', '#C73E1D']
width = 0.25

# Plot 1: Worker Characteristics
for i in range(N):
    x_pos = i
    ax1.bar(x_pos - width, q_i[i], width, color=colors[0], edgecolor='black', linewidth=0.5)
    ax1.bar(x_pos, p_i[i], width, color=colors[1], edgecolor='black', linewidth=0.5)
    ax1.bar(x_pos + width, epsilon_i[i], width, color=colors[2], edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Worker ID', fontweight='bold')
ax1.set_ylabel('Value', fontweight='bold')
ax1.set_title('(a) Worker Characteristics (10 Workers)')  # Updated title
ax1.set_xticks(np.arange(N))
ax1.set_xticklabels(workers)
ax1.legend(['Optimal $q_i^*$', 'CPU Power $p_i$', 'Accuracy $\epsilon_i$'],
            frameon=True, edgecolor='black', ncol=3)
ax1.grid(True, axis='y')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

# Plot 2: Utility Decomposition
for i in range(N):
    # Plot reward (positive) separately from costs (negative)
    ax2.bar(i+1, reward[i], width=0.6, color=colors[0],
            label='Reward' if i == 0 else "", edgecolor='black', linewidth=0.5)

    # Plot costs stacked below zero
    bottom = 0
    for j, (comp, label) in enumerate(zip(
        [energy_cost[i], distance_cost[i]],
        ['Energy Cost', 'Distance Cost']
    )):
        ax2.bar(i+1, comp, width=0.6, bottom=bottom, color=colors[j+1],
                label=label if i == 0 else "", edgecolor='black', linewidth=0.5)
        bottom += comp

ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_xlabel('Worker ID', fontweight='bold')
ax2.set_ylabel('Utility Components', fontweight='bold')
ax2.set_title('(b) Utility Decomposition (10 Workers)')  # Updated title
ax2.set_xticks(workers)
ax2.legend(frameon=True, edgecolor='black', loc='upper right')
ax2.grid(True, axis='y')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('worker_analysis_10tasks.png', bbox_inches='tight', dpi=300)  # Updated filename
plt.show()

# Table output (all 10 workers)
headers = ["Worker", "$p_i$", "$\epsilon_i$", "$q_i^*$", "Reward", "Energy Cost", "Distance Cost", "Utility", "Budget Share"]
table_data = []

for i in range(N):
    budget_share = reward[i]
    table_data.append([
        f"{i+1}",
        f"{p_i[i]:.4f}",
        f"{epsilon_i[i]:.4f}",
        f"{q_i[i]:.4f}",
        f"{reward[i]:.4f}",
        f"{energy_cost[i]:.4f}",
        f"{distance_cost[i]:.4f}",
        f"{U_i[i]:.4f}",
        f"{budget_share:.4f}"
    ])

# Add summary rows
table_data.append([
    "Mean",
    f"{np.mean(p_i):.4f}",
    f"{np.mean(epsilon_i):.4f}",
    f"{np.mean(q_i):.4f}",
    f"{np.mean(reward):.4f}",
    f"{np.mean(energy_cost):.4f}",
    f"{np.mean(distance_cost):.4f}",
    f"{np.mean(U_i):.4f}",
    f"{total_budget_share:.4f}/{B}"
])
table_data.append([
    "Std Dev",
    f"{np.std(p_i):.4f}",
    f"{np.std(epsilon_i):.4f}",
    f"{np.std(q_i):.4f}",
    f"{np.std(reward):.4f}",
    f"{np.std(energy_cost):.4f}",
    f"{np.std(distance_cost):.4f}",
    f"{np.std(U_i):.4f}",
    "-"
])

print("NUMERICAL RESULTS (10 WORKERS)\n" + "="*60)  # Updated header
print(tabulate(table_data, headers=headers, tablefmt="grid"))
print(f"\nBudget Status: {'Within budget' if total_budget_share <= B else 'Over budget'} ({total_budget_share:.4f}/{B})")

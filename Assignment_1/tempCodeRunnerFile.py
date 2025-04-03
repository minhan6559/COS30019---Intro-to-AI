import matplotlib.pyplot as plt
import numpy as np

# Data from the results
algorithms = ["BFS", "DFS", "A*", "Greedy", "UCS", "BULB"]
path_costs = [25676.09, 422484.24, 22011.55, 26701.98, 22011.55, 22011.55]
path_lengths = [32, 669, 42, 37, 42, 42]
nodes_expanded = [8393, 3708, 1318, 37, 9152, 13317]
runtimes = [2.2889, 17.0516, 0.0968, 0.0035, 0.4339, 0.1661]

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Search Algorithm Performance Comparison", fontsize=16)

# Colors for the bars
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

# Plot 1: Path Cost (log scale due to large range)
axs[0, 0].bar(algorithms, path_costs, color=colors)
axs[0, 0].set_title("Path Cost")
axs[0, 0].set_yscale("log")
axs[0, 0].set_ylabel("Cost (log scale)")
axs[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

# Plot 2: Path Length (log scale due to DFS's large value)
axs[0, 1].bar(algorithms, path_lengths, color=colors)
axs[0, 1].set_title("Path Length")
axs[0, 1].set_yscale("log")
axs[0, 1].set_ylabel("Length (log scale)")
axs[0, 1].grid(axis="y", linestyle="--", alpha=0.7)

# Plot 3: Nodes Expanded (log scale)
axs[1, 0].bar(algorithms, nodes_expanded, color=colors)
axs[1, 0].set_title("Nodes Expanded")
axs[1, 0].set_yscale("log")
axs[1, 0].set_ylabel("Count (log scale)")
axs[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

# Plot 4: Runtime
axs[1, 1].bar(algorithms, runtimes, color=colors)
axs[1, 1].set_title("Runtime")
axs[1, 1].set_ylabel("Time (seconds)")
axs[1, 1].set_yscale("log")  # Log scale because of large differences
axs[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

# Add value labels on top of each bar
for ax in axs.flat:
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if ax == axs[0, 0]:  # Path Cost - round to 2 decimal places
            value = f"{path_costs[i]:.2f}"
        elif ax == axs[0, 1]:  # Path Length - integer
            value = f"{path_lengths[i]}"
        elif ax == axs[1, 0]:  # Nodes Expanded - integer
            value = f"{nodes_expanded[i]}"
        else:  # Runtime - round to 4 decimal places
            value = f"{runtimes[i]:.4f}"

        # Position the text above the bar
        ax.text(
            p.get_x() + p.get_width() / 2,
            height * 1.02,
            value,
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=9,
        )

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("search_algorithm_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

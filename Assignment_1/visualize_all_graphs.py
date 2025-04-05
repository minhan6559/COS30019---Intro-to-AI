import os
import matplotlib.pyplot as plt
import numpy as np
from src.parser.graph_parser import GraphParser


def visualize_and_save_graph(filepath, output_dir):
    """
    Parse a graph file, visualize the graph, and save it as an image.

    Args:
        filepath: Path to the graph file
        output_dir: Directory to save the output image

    Returns:
        Path to the saved image
    """
    # Parse the graph file
    parser = GraphParser()
    parser.parse_file(filepath)
    graph, origin, destinations, locations = parser.get_problem_components()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot nodes
    for node_id, (x, y) in locations.items():
        # Use different colors for origin and destinations
        if node_id == origin:
            node_color = "green"  # Origin node in green
            node_size = 600
        elif node_id in destinations:
            node_color = "red"  # Destination nodes in red
            node_size = 600
        else:
            node_color = "skyblue"  # Regular nodes in blue
            node_size = 600

        # Plot the node
        ax.scatter(x, y, s=node_size, c=node_color, edgecolors="black", zorder=5)

        # Place label directly on the node (centered)
        ax.text(
            x,
            y,
            str(node_id),
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=10,
            color="black",
            zorder=6,
        )

    # Plot edges with arrows
    for source, targets in graph.graph_dict.items():
        for target, weight in targets.items():
            if source in locations and target in locations:
                # Get coordinates
                x1, y1 = locations[source]
                x2, y2 = locations[target]

                # Calculate arrow properties
                dx = x2 - x1
                dy = y2 - y1

                # Draw arrow (slightly shorter to not overlap with nodes)
                ax.arrow(
                    x1,
                    y1,
                    dx * 0.85,
                    dy * 0.85,
                    head_width=0.03,
                    head_length=0.05,
                    fc="black",
                    ec="black",
                    length_includes_head=True,
                    zorder=1,
                )

                # Add weight label at the middle of the edge
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                offset_x = 0
                offset_y = 0
                ax.text(
                    mid_x + offset_x,
                    mid_y + offset_y,
                    f"{weight}",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
                    zorder=2,
                )

    # Set axis properties
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(f"Graph Visualization - {os.path.basename(filepath)}")

    # Set margins to ensure all nodes are visible
    plt.margins(0.2)
    plt.tight_layout()

    # Add legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label="Origin Node",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="Destination Node",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="skyblue",
            markersize=10,
            label="Regular Node",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # Add grid
    plt.grid(True, linestyle="--", alpha=0.7)

    # Create output filename and save
    filename = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory

    return output_path


def main():
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(base_dir, "testcases")
    output_dir = os.path.join(testcases_dir, "visualization")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Get all txt files in testcases directory
    txt_files = [f for f in os.listdir(testcases_dir) if f.endswith(".txt")]

    if not txt_files:
        print("No .txt files found in the testcases directory.")
        return

    print(f"Found {len(txt_files)} test case files.")

    # Process each file
    for i, file in enumerate(txt_files):
        file_path = os.path.join(testcases_dir, file)
        print(f"[{i+1}/{len(txt_files)}] Processing {file}...")
        try:
            output_path = visualize_and_save_graph(file_path, output_dir)
            print(f"    Saved visualization to {output_path}")
        except Exception as e:
            print(f"    Error processing {file}: {str(e)}")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

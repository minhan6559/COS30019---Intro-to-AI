from src.problem.multigoal_graph_problem import MultigoalGraphProblem
import os


def main():
    """Example usage of GraphProblem's file loading capabilities"""

    # You can use different ways to specify the file path:

    # 1. Just the filename (will be searched in multiple locations)
    filename = "exported_graph.txt"

    print(f"Loading graph from file: {filename}")

    # Create the problem directly from the file
    problem = MultigoalGraphProblem.from_file(filename)

    print("\nGraph Problem Information:")
    print("--------------------------")
    print(problem)  # This uses the __repr__ method

    print("\nDetailed Graph Data:")
    print("-------------------")
    print("Starting point:", problem.initial)
    print("Goals:", problem.goals)

    print("\nGraph Structure:")
    for src, neighbors in problem.graph.graph_dict.items():
        print(f"{src} -> {neighbors}")

    print("\nNode Coordinates:")
    for node, coords in problem.locations.items():
        print(f"{node}: {coords}")

    print("\nGraph Statistics:")
    print(f"Number of nodes: {len(problem.graph.nodes())}")
    print(
        f"Number of edges: {sum(len(neighbors) for neighbors in problem.graph.graph_dict.values())}"
    )

    # Export the graph to a file
    export_filename = "exported_graph.txt"
    MultigoalGraphProblem.to_file(problem, export_filename)
    print(f"\nGraph exported to {export_filename}")

    ## Load the graph from the exported file and verify it
    loaded_problem = MultigoalGraphProblem.from_file(export_filename)
    print("\nLoaded Graph Problem Information:")
    print("--------------------------")
    print(loaded_problem)  # This uses the __repr__ method

    return 0


if __name__ == "__main__":
    main()

from src.problem.graph_problem import GraphProblem
import os


def main():
    """Example usage of GraphProblem's file loading capabilities"""

    # You can use different ways to specify the file path:

    # 1. Just the filename (will be searched in multiple locations)
    filename = "testcases/testcase3.txt"

    print(f"Loading graph from file: {filename}")

    # Create the problem directly from the file
    problem = GraphProblem.from_file(filename)

    print("\nGraph Problem Information:")
    print("--------------------------")
    print(problem)  # This uses the __repr__ method

    print("\nDetailed Graph Data:")
    print("-------------------")
    print("Starting point:", problem.initial)
    print("Goals:", problem.goals)
    print("Current goal:", problem.current_goal)

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

    return 0


if __name__ == "__main__":
    main()

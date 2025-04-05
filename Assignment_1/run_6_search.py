from src.problem.multigoal_graph_problem import MultigoalGraphProblem
from src.search_algorithm.search_algorithm import (
    BreadthFirstSearch,
    DepthFirstSearch,
    AStarSearch,
    GreedyBestFirstSearch,
    UniformCostSearch,
    BULBSearch,
)
import time
import sys


def main():
    """Example of creating and solving a random graph problem"""

    show_path = False  # Set to False to not show the path in the output

    # Check if correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python run_6_search.py <filename>")
        return 1

    # Parse command-line arguments
    filename = sys.argv[1]
    # Load the graph problem from the file
    try:
        original_problem = MultigoalGraphProblem.from_file(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    # Store original problem data we'll need for creating fresh instances
    original_initial = original_problem.initial
    original_goals = original_problem.goals.copy()
    original_graph = original_problem.graph
    original_locations = original_problem.locations

    # Dictionary to store search results
    all_results = {}

    # Set up available search algorithms
    algorithms = {
        "BFS": BreadthFirstSearch(),
        "DFS": DepthFirstSearch(),
        "A*": AStarSearch(),
        "Greedy": GreedyBestFirstSearch(),
        "UCS": UniformCostSearch(),
        "BULB": BULBSearch(),
    }

    # First, run the multi-goal search
    print(f"\n{'=' * 50}")
    print(f"Searching paths for multi-goal problem (all destinations):")
    print(f"{'=' * 50}")

    # Create a fresh instance of the multi-goal problem
    multi_goal_problem = MultigoalGraphProblem(
        original_initial,
        original_goals,  # Pass the list of all goals
        original_graph,
        original_locations,
    )

    # Dictionary to store multi-goal results
    all_results["multi-goal"] = {}

    # Run each algorithm against the multi-goal problem
    for name, algorithm in algorithms.items():
        start_time = time.time()

        # The result now returns a tuple (node, expanded_count, created_count)
        result_tuple = algorithm.search(multi_goal_problem)

        elapsed_time = time.time() - start_time

        if result_tuple and result_tuple[0]:  # Check if node exists
            result_node = result_tuple[0]
            expanded_count = result_tuple[1]
            created_count = result_tuple[2]

            result_state = result_node.state
            path_nodes = result_node.path_states()
            path_cost = result_node.path_cost

            # Store in results
            all_results["multi-goal"][name] = {
                "Destination": result_state,
                "path": path_nodes,
                "cost": path_cost,
                "time": elapsed_time,
                "expanded": expanded_count,
                "created": created_count,
            }
        else:
            expanded_count = result_tuple[1] if result_tuple else 0
            created_count = result_tuple[2] if result_tuple else 0
            print(f"  {name}: No path found after {elapsed_time:.4f} seconds")
            print(f"  Nodes expanded: {expanded_count}, Nodes created: {created_count}")

            all_results["multi-goal"][name] = {
                "Destination": "N/A",
                "path": None,
                "cost": float("inf"),
                "time": elapsed_time,
                "expanded": expanded_count,
                "created": created_count,
            }

    # Print performance comparison
    print("\n" + "=" * 60)
    print("Performance Comparison Across All Problems")
    print("=" * 60)

    # First print multi-goal results
    print("\nMulti-Goal Problem:")
    print(f"{'-' * 100}")
    print(
        f"{'Algorithm':<10} {'Success':<8} {'Dest':<8} {'Path Cost':<12} {'Path Length':<12} {'Expanded':<10} {'Created':<10} {'Time (s)':<10} {'Path':<10}"
    )
    print(f"{'-' * 100}")

    for name, result in all_results["multi-goal"].items():
        success = "Yes" if result["path"] is not None else "No"
        destination = result["Destination"]
        cost = f"{result['cost']:.2f}" if result["path"] is not None else "N/A"
        path_len = len(result["path"]) if result["path"] is not None else "N/A"
        expanded = f"{result['expanded']}"
        created = f"{result['created']}"
        time_taken = f"{result['time']:.4f}"
        path_nodes = f"{result['path']}" if result["path"] is not None else "N/A"

        print(
            f"{name:<10} {success:<8} {destination:<8} {cost:<12} {path_len:<12} {expanded:<10} {created:<10} {time_taken:<10} {path_nodes:<10}"
        )


if __name__ == "__main__":
    main()

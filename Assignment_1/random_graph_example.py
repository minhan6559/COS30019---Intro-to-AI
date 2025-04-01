from src.problem.graph_problem import GraphProblem
from src.search_algorithm.search_algorithm import (
    BreadthFirstSearch,
    DepthFirstSearch,
    AStarSearch,
    GreedyBestFirstSearch,
    DijkstraSearch,
    IDAStarSearch,
    # BidirectionalAStarSearch,
)
import time


def main():
    """Example of creating and solving a random graph problem"""

    show_path = False  # Set to False to not show the path in the output

    # Parameters for graph
    num_nodes = 600
    min_edges_per_node = 2
    max_edges_per_node = 5
    grid_size = 2000
    num_destinations = 2
    ensure_connectivity = True

    print("Generating random graph problem...")
    print(f"Number of nodes: {num_nodes}")
    print(f"Edges per node: {min_edges_per_node}-{max_edges_per_node}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Number of destinations: {num_destinations}")

    # Create a random graph problem
    start_gen_time = time.time()
    original_problem = GraphProblem.random(
        num_nodes=num_nodes,
        min_edges_per_node=min_edges_per_node,
        max_edges_per_node=max_edges_per_node,
        grid_size=grid_size,
        num_destinations=num_destinations,
        ensure_connectivity=ensure_connectivity,
    )
    gen_time = time.time() - start_gen_time
    print(f"Graph generation time: {gen_time:.4f} seconds")

    print("\nRandom graph problem created:")
    print(original_problem)

    print(f"\nStarting point: {original_problem.initial}")
    print(f"Goals: {original_problem.goals}")

    # Print a sample of nodes and their connections
    print("\nSample of graph connections:")
    sample_nodes = list(original_problem.graph.graph_dict.keys())[:5]  # First 5 nodes
    for node in sample_nodes:
        neighbors = original_problem.graph.get(node)
        print(f"Node {node} connects to: {neighbors}")

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
        "Dijkstra": DijkstraSearch(),
        "IDA*": IDAStarSearch(),
        # "Bidirectional": BidirectionalAStarSearch(),
    }

    # For each goal, create a fresh problem instance and run each search algorithm
    for goal in original_goals:
        print(f"\n{'=' * 50}")
        print(f"Searching paths to destination {goal}:")
        print(f"{'=' * 50}")

        # Create a fresh problem instance for this goal
        # This ensures no memoization side effects between different goals
        fresh_problem = GraphProblem(
            original_initial,
            goal,  # Pass single goal instead of list
            original_graph,
            original_locations,
        )

        # Run each algorithm against the current goal
        for name, algorithm in algorithms.items():
            print(f"\nRunning {name}...")
            start_time = time.time()

            result = algorithm.search(fresh_problem)

            elapsed_time = time.time() - start_time

            if result:
                path = result.path()
                path_nodes = [node.state for node in path]
                path_cost = result.path_cost

                if show_path:
                    print(f"  {name}: Path found in {elapsed_time:.4f} seconds")
                    print(f"  Cost: {path_cost:.2f}, Path length: {len(path_nodes)}")
                    print(f"  Path: {' -> '.join(map(str, path_nodes))}")

                # Store in results
                if goal not in all_results:
                    all_results[goal] = {}
                all_results[goal][name] = {
                    "path": path_nodes,
                    "cost": path_cost,
                    "time": elapsed_time,
                }
            else:
                print(f"  {name}: No path found after {elapsed_time:.4f} seconds")

                if goal not in all_results:
                    all_results[goal] = {}
                all_results[goal][name] = {
                    "path": None,
                    "cost": float("inf"),
                    "time": elapsed_time,
                }

    # Print performance comparison
    print("\n" + "=" * 60)
    print("Performance Comparison Across All Destinations")
    print("=" * 60)

    for goal, algorithms_results in all_results.items():
        print(f"\nDestination {goal}:")
        print(f"{'-' * 50}")
        print(
            f"{'Algorithm':<10} {'Success':<8} {'Path Cost':<12} {'Path Length':<14} {'Time (s)':<10}"
        )
        print(f"{'-' * 50}")

        for name, result in algorithms_results.items():
            success = "Yes" if result["path"] is not None else "No"
            cost = f"{result['cost']:.2f}" if result["path"] is not None else "N/A"
            path_len = len(result["path"]) if result["path"] is not None else "N/A"
            time_taken = f"{result['time']:.4f}"

            print(f"{name:<10} {success:<8} {cost:<12} {path_len:<14} {time_taken:<10}")


if __name__ == "__main__":
    main()

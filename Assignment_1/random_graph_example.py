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


def main():
    """Example of creating and solving a random graph problem"""

    show_path = False  # Set to False to not show the path in the output

    # Parameters for graph
    num_nodes = 10000
    min_edges_per_node = 10
    max_edges_per_node = 20
    grid_size = 25000
    num_destinations = 3
    ensure_connectivity = True

    print("Generating random graph problem...")
    print(f"Number of nodes: {num_nodes}")
    print(f"Edges per node: {min_edges_per_node}-{max_edges_per_node}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Number of destinations: {num_destinations}")

    # Create a random graph problem
    start_gen_time = time.time()
    original_problem = MultigoalGraphProblem.random(
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
        "BULB": BULBSearch(beam_width=6, max_discrepancies=10),
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
        print(f"\nRunning {name} on multi-goal problem...")
        start_time = time.time()

        # The result now returns a tuple (node, expanded_count)
        result_tuple = algorithm.search(multi_goal_problem)

        elapsed_time = time.time() - start_time

        if result_tuple and result_tuple[0]:  # Check if node exists
            result_node = result_tuple[0]
            expanded_count = result_tuple[1]

            result_state = result_node.state
            path_nodes = result_node.path_states()
            path_cost = result_node.path_cost

            if show_path:
                print(f"  {name}: Path found in {elapsed_time:.4f} seconds")
                print(f"  Cost: {path_cost:.2f}, Path length: {len(path_nodes)}")
                print(f"  Nodes expanded: {expanded_count}")
                print(f"  Path: {' -> '.join(map(str, path_nodes))}")

            # Store in results
            all_results["multi-goal"][name] = {
                "Destination": result_state,
                "path": path_nodes,
                "cost": path_cost,
                "time": elapsed_time,
                "expanded": expanded_count,
            }
        else:
            expanded_count = result_tuple[1] if result_tuple else 0
            print(f"  {name}: No path found after {elapsed_time:.4f} seconds")
            print(f"  Nodes expanded: {expanded_count}")

            all_results["multi-goal"][name] = {
                "Destination": "N/A",
                "path": None,
                "cost": float("inf"),
                "time": elapsed_time,
                "expanded": expanded_count,
            }

    # For each individual goal, create a fresh problem instance and run each search algorithm
    for goal in original_goals:
        print(f"\n{'=' * 50}")
        print(f"Searching paths to single destination {goal}:")
        print(f"{'=' * 50}")

        # Create a fresh problem instance for this goal
        fresh_problem = MultigoalGraphProblem(
            original_initial,
            goal,  # Pass single goal instead of list
            original_graph,
            original_locations,
        )

        # Run each algorithm against the current goal
        for name, algorithm in algorithms.items():
            print(f"\nRunning {name}...")
            start_time = time.time()

            # The result now returns a tuple (node, expanded_count)
            result_tuple = algorithm.search(fresh_problem)

            elapsed_time = time.time() - start_time

            if result_tuple and result_tuple[0]:  # Check if node exists
                result_node = result_tuple[0]
                expanded_count = result_tuple[1]

                path_nodes = result_node.path_states()
                path_cost = result_node.path_cost

                if show_path:
                    print(f"  {name}: Path found in {elapsed_time:.4f} seconds")
                    print(f"  Cost: {path_cost:.2f}, Path length: {len(path_nodes)}")
                    print(f"  Nodes expanded: {expanded_count}")
                    print(f"  Path: {' -> '.join(map(str, path_nodes))}")

                # Store in results
                if goal not in all_results:
                    all_results[goal] = {}
                all_results[goal][name] = {
                    "path": path_nodes,
                    "cost": path_cost,
                    "time": elapsed_time,
                    "expanded": expanded_count,
                }
            else:
                expanded_count = result_tuple[1] if result_tuple else 0
                print(f"  {name}: No path found after {elapsed_time:.4f} seconds")
                print(f"  Nodes expanded: {expanded_count}")

                if goal not in all_results:
                    all_results[goal] = {}
                all_results[goal][name] = {
                    "path": None,
                    "cost": float("inf"),
                    "time": elapsed_time,
                    "expanded": expanded_count,
                }

    # Print performance comparison
    print("\n" + "=" * 60)
    print("Performance Comparison Across All Problems")
    print("=" * 60)

    # First print multi-goal results
    print("\nMulti-Goal Problem:")
    print(f"{'-' * 80}")
    print(
        f"{'Algorithm':<10} {'Success':<8} {'Dest':<8} {'Path Cost':<12} {'Path Length':<14} {'Expanded':<10} {'Time (s)':<10}"
    )
    print(f"{'-' * 80}")

    for name, result in all_results["multi-goal"].items():
        success = "Yes" if result["path"] is not None else "No"
        destination = result["Destination"]
        cost = f"{result['cost']:.2f}" if result["path"] is not None else "N/A"
        path_len = len(result["path"]) if result["path"] is not None else "N/A"
        expanded = f"{result['expanded']}"
        time_taken = f"{result['time']:.4f}"

        print(
            f"{name:<10} {success:<8} {destination:<8} {cost:<12} {path_len:<14} {expanded:<10} {time_taken:<10}"
        )

    # Then print individual goal results
    for goal, algorithms_results in all_results.items():
        if goal == "multi-goal":
            continue  # Skip the multi-goal results as we've already printed them

        print(f"\nDestination {goal}:")
        print(f"{'-' * 70}")
        print(
            f"{'Algorithm':<10} {'Success':<8} {'Path Cost':<12} {'Path Length':<14} {'Expanded':<10} {'Time (s)':<10}"
        )
        print(f"{'-' * 70}")

        for name, result in algorithms_results.items():
            success = "Yes" if result["path"] is not None else "No"
            cost = f"{result['cost']:.2f}" if result["path"] is not None else "N/A"
            path_len = len(result["path"]) if result["path"] is not None else "N/A"
            expanded = f"{result['expanded']}"
            time_taken = f"{result['time']:.4f}"

            print(
                f"{name:<10} {success:<8} {cost:<12} {path_len:<14} {expanded:<10} {time_taken:<10}"
            )


if __name__ == "__main__":
    main()

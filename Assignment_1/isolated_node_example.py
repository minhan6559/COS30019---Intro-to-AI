from src.problem.multigoal_graph_problem import MultigoalGraphProblem
from src.search_algorithm.search_algorithm import (
    BreadthFirstSearch,
    DepthFirstSearch,
    AStarSearch,
    GreedyBestFirstSearch,
    UniformCostSearch,
    BULBSearch,
)
from src.graph.graph import Graph
from src.utils.utils import distance
import time
import copy


def main():
    """Example of creating a random graph problem with an isolated destination"""

    show_path = False  # Set to False to not show the path in the output

    # Parameters for graph
    num_nodes = 10000
    min_edges_per_node = 5
    max_edges_per_node = 8
    grid_size = 25000
    num_destinations = 3  # We need exactly 3 destinations
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

    # Find the closest destination to the initial node (A)
    initial_node = original_problem.initial
    destinations = original_problem.goals
    locations = original_problem.locations

    closest_dest = None
    min_distance = float("inf")
    for dest in destinations:
        dist = distance(locations[initial_node], locations[dest])
        if dist < min_distance:
            min_distance = dist
            closest_dest = dest

    print(
        f"\nClosest destination to initial node: {closest_dest} (distance: {min_distance:.2f})"
    )
    print(f"Isolating node {closest_dest} (removing all connections)...")

    # Create a deep copy of the graph dictionary to modify
    graph_dict = copy.deepcopy(original_problem.graph.graph_dict)

    # Count connections to be removed
    connection_count = 0
    for node in graph_dict:
        if closest_dest in graph_dict[node]:
            connection_count += 1

    # Also count outgoing connections from closest_dest
    if closest_dest in graph_dict:
        connection_count += len(graph_dict[closest_dest])

    # Isolate the closest destination by removing all edges to/from it
    for node in graph_dict:
        if closest_dest in graph_dict[node]:
            # Remove edge from node to closest_dest
            del graph_dict[node][closest_dest]

    # Remove all outgoing edges from closest_dest
    if closest_dest in graph_dict:
        graph_dict[closest_dest] = {}

    print(f"Removed {connection_count} connections to/from node {closest_dest}")

    # Create a new problem instance with the modified graph
    isolated_problem = MultigoalGraphProblem(
        original_problem.initial,
        original_problem.goals,
        Graph(graph_dict),
        original_problem.locations,
    )

    # Store data we'll need for creating fresh instances
    initial = isolated_problem.initial
    goals = isolated_problem.goals.copy()
    graph = isolated_problem.graph
    locations = isolated_problem.locations

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
        initial,
        goals,  # Pass the list of all goals
        graph,
        locations,
    )

    # Dictionary to store multi-goal results
    all_results["multi-goal"] = {}

    # Run each algorithm against the multi-goal problem
    for name, algorithm in algorithms.items():
        print(f"\nRunning {name} on multi-goal problem...")
        start_time = time.time()

        result = algorithm.search(multi_goal_problem)

        elapsed_time = time.time() - start_time

        if result:
            result_state = result.state
            path = result.path()
            path_nodes = [node.state for node in path]
            path_cost = result.path_cost

            if show_path:
                print(f"  {name}: Path found in {elapsed_time:.4f} seconds")
                print(f"  Cost: {path_cost:.2f}, Path length: {len(path_nodes)}")
                print(f"  Path: {' -> '.join(map(str, path_nodes))}")

            # Store in results
            all_results["multi-goal"][name] = {
                "Destination": result_state,
                "path": path_nodes,
                "cost": path_cost,
                "time": elapsed_time,
            }
        else:
            print(f"  {name}: No path found after {elapsed_time:.4f} seconds")

            all_results["multi-goal"][name] = {
                "Destination": "N/A",
                "path": None,
                "cost": float("inf"),
                "time": elapsed_time,
            }

    # For each individual goal, create a fresh problem instance and run each search algorithm
    for goal in goals:
        print(f"\n{'=' * 50}")
        print(f"Searching paths to single destination {goal}:")
        print(f"{'=' * 50}")

        # Create a fresh problem instance for this goal
        fresh_problem = MultigoalGraphProblem(
            initial,
            goal,  # Pass single goal instead of list
            graph,
            locations,
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
    print("Performance Comparison Across All Problems")
    print("=" * 60)

    # First print multi-goal results
    print("\nMulti-Goal Problem:")
    print(f"{'-' * 50}")
    print(
        f"{'Algorithm':<10} {'Success':<8} {'Dest':<8} {'Path Cost':<12} {'Path Length':<14} {'Time (s)':<10}"
    )
    print(f"{'-' * 50}")

    for name, result in all_results["multi-goal"].items():
        success = "Yes" if result["path"] is not None else "No"
        destination = result["Destination"]
        cost = f"{result['cost']:.2f}" if result["path"] is not None else "N/A"
        path_len = len(result["path"]) if result["path"] is not None else "N/A"
        time_taken = f"{result['time']:.4f}"

        print(
            f"{name:<10} {success:<8} {destination:<8} {cost:<12} {path_len:<14} {time_taken:<10}"
        )

    # Then print individual goal results
    for goal, algorithms_results in all_results.items():
        if goal == "multi-goal":
            continue  # Skip the multi-goal results as we've already printed them

        print(f"\nDestination {goal}:")
        print(f"{'-' * 50}")
        print(
            f"{'Algorithm':<10} {'Success':<8} {'Path Cost':<12} {'Path Length':<14} {'Time (s)':<10}"
        )
        print(f"{'-' * 50}")

        # Highlight if this is the isolated node
        is_isolated = goal == closest_dest
        if is_isolated:
            print(f"NOTE: This is the isolated destination node (Node A)")

        for name, result in algorithms_results.items():
            success = "Yes" if result["path"] is not None else "No"
            cost = f"{result['cost']:.2f}" if result["path"] is not None else "N/A"
            path_len = len(result["path"]) if result["path"] is not None else "N/A"
            time_taken = f"{result['time']:.4f}"

            print(f"{name:<10} {success:<8} {cost:<12} {path_len:<14} {time_taken:<10}")


if __name__ == "__main__":
    main()

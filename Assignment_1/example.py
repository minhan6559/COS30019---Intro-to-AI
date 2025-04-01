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


def main():
    """Example of using GraphProblem with multiple goals"""

    # Example graph file
    graph_file = "testcases/testcase3.txt"  # Simple filename, will be searched in multiple locations

    print(f"Loading graph file: {graph_file}")

    # Create a multi-goal problem directly using the from_file method
    problem = GraphProblem.from_file(graph_file)

    print(f"\nCreated graph problem:")
    print(problem)  # This will use the __repr__ method

    print(f"Starting point: {problem.initial}")
    print(f"Goals: {problem.goals}")
    print(f"Current goal: {problem.current_goal}")

    # Create a dictionary to store search results for each algorithm
    all_results = {}

    # Available search algorithms
    algorithms = {
        "BFS": BreadthFirstSearch(),
        "DFS": DepthFirstSearch(),
        "A*": AStarSearch(),
        "Greedy": GreedyBestFirstSearch(),
        "Dijkstra": DijkstraSearch(),
        "IDA*": IDAStarSearch(),
        # "Bidirectional": BidirectionalAStarSearch(),
    }

    # For each goal in the problem, run each algorithm
    original_goals = problem.goals.copy()

    for _ in range(len(original_goals)):
        current_goal = problem.current_goal
        print(f"\nSearching paths to destination {current_goal}:")
        print(f"Problem: {problem}")  # Print the problem representation

        # Run each algorithm against the current goal
        for name, algorithm in algorithms.items():
            result = algorithm.search(problem)

            if result:
                path = result.path()
                path_nodes = [node.state for node in path]
                path_cost = result.path_cost

                print(f"  {name}: Path found with cost {path_cost}")
                print(f"    Path: {' -> '.join(map(str, path_nodes))}")

                # Store in results
                if current_goal not in all_results:
                    all_results[current_goal] = {}
                all_results[current_goal][name] = {
                    "path": path_nodes,
                    "cost": path_cost,
                }
            else:
                print(f"  {name}: No path found")

        # Switch to the next goal for the next iteration
        problem.next_goal()

    # Compare the results
    print("\nResults summary:")
    for goal, algorithms_results in all_results.items():
        print(f"\nDestination {goal}:")
        for name, result in algorithms_results.items():
            print(
                f"  {name}: Cost = {result['cost']}, Path length = {len(result['path'])}"
            )


if __name__ == "__main__":
    main()

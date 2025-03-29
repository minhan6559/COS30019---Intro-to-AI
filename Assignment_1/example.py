from src.parser.graph_parser import create_graph_problem_from_file
from src.problem.graph_problem import GraphProblem
from src.search_algorithm.search_algorithm import (
    BreadthFirstSearch,
    DepthFirstSearch,
    AStarSearch,
    GreedyBestFirstSearch,
    DijkstraSearch,
)


def main():
    """Example of using the parser with search algorithms"""

    # Example of different ways to specify the graph file
    graph_file = "testcases/testcase10.txt"  # Simple filename, will be searched in multiple locations

    # Alternative ways to specify the path:
    # - Absolute path
    # absolute_path = os.path.abspath("PathFinder-test.txt")
    # - Path relative to testcases directory
    # testcase_path = os.path.join("testcases", "PathFinder-test.txt")

    print(f"Attempting to parse graph file: {graph_file}")

    # 1. Load the graph from a file
    graph, origin, destinations, locations = create_graph_problem_from_file(graph_file)

    print(f"\nLoaded graph with {len(graph.nodes())} nodes")
    print(f"Origin: {origin}")
    print(f"Destinations: {destinations}")

    # Create a dictionary to store search results for each algorithm
    all_results = {}

    # 2. Run different search algorithms
    algorithms = {
        "BFS": BreadthFirstSearch(),
        "DFS": DepthFirstSearch(),
        "A*": AStarSearch(),
        "Greedy": GreedyBestFirstSearch(),
        "Dijkstra": DijkstraSearch(),
    }

    # For each destination, run each algorithm
    for destination in destinations:
        print(f"\nSearching paths to destination {destination}:")

        # Create a problem for this destination
        problem = GraphProblem(origin, destination, graph, locations)

        # Run each algorithm
        for name, algorithm in algorithms.items():
            result = algorithm.search(problem)

            if result:
                path = result.path()
                path_nodes = [node.state for node in path]
                path_cost = result.path_cost

                print(f"  {name}: Path found with cost {path_cost}")
                print(f"    Path: {' -> '.join(map(str, path_nodes))}")

                # Store in results
                if destination not in all_results:
                    all_results[destination] = {}
                all_results[destination][name] = {"path": path_nodes, "cost": path_cost}
            else:
                print(f"  {name}: No path found")

    # 3. Compare the results
    print("\nResults summary:")
    for destination, algorithms_results in all_results.items():
        print(f"\nDestination {destination}:")
        for name, result in algorithms_results.items():
            print(
                f"  {name}: Cost = {result['cost']}, Path length = {len(result['path'])}"
            )


if __name__ == "__main__":
    main()

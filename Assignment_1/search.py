import sys

from src.problem.multigoal_graph_problem import MultigoalGraphProblem
from src.search_algorithm.search_algorithm import *




def main():
    # Check if correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        print("Available methods: BFS, DFS, A*, GBFS, ..., ...")
        return 1

    # Parse command-line arguments
    filename = sys.argv[1]
    method = sys.argv[2].upper()

    # Map method strings to search algorithm objects
    methods = {
        "BFS": BreadthFirstSearch(),
        "DFS": DepthFirstSearch(),
        "AS": AStarSearch(),
        "GBFS": GreedyBestFirstSearch(),
        "CUS1": UniformCostSearch(),
        "CUS2": BULBSearch(),
    }

    # Check if the method is valid
    if method not in methods:
        print(f"Error: Unknown method '{method}'")
        print("Available methods: BFS, DFS, A*, GBFS, UCS, BULB")
        return 1

    try:
        # Load the graph problem from the file
        problem = MultigoalGraphProblem.from_file(filename)
        
        # Use the first goal in the list
        if problem.goals:
            problem.current_goal = problem.goals[0]
        
        # Execute the search algorithm
        search_algorithm = methods[method]
        result_tuple = search_algorithm.search(problem)
        
        # Output the results in the required format
        if result_tuple and result_tuple[0]:
            result_node = result_tuple[0]
            expanded_count = result_tuple[1]
            nodes_created = result_tuple[2]
            print("\n")
            # First line: filename method
            print(f"{filename} {method}")
            # Second line: goal number_of_nodes
            print(f"{problem.current_goal} {nodes_created}")
            
            # Third line: path
            path = [node.state for node in result_tuple[0].path_nodes()]
            print(" ".join(map(str, path)))
            print("\n")
            return 0  # Success
        else:
            print(f"No solution found for {filename} using {method}")
            return 1  # No solution
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
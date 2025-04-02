# import sys
# from src.problem.graph_problem import GraphProblem
# from src.search_algorithm.search_algorithm import (
#     BreadthFirstSearch,
#     DepthFirstSearch,
#     AStarSearch,
#     GreedyBestFirstSearch,
#     # DijkstraSearch,
# )
# # Add a static counter to the Node class for tracking node creation
# from src.graph.graph import Node


# Node.node_count = 0  # Initialize static counter
# original_init = Node.__init__  # Store the original __init__ method

# # Override the Node.__init__ method to count nodes
# def counting_init(self, state, parent=None, action=None, path_cost=0):
#     # Increment the counter
#     Node.node_count += 1
#     # Call the original __init__ method
#     original_init(self, state, parent, action, path_cost)

# # Replace the original __init__ with our counting version
# Node.__init__ = counting_init

# def main():
#     # Check if correct number of arguments
#     if len(sys.argv) != 3:
#         print("Usage: python search.py <filename> <method>")
#         print("Available methods: BFS, DFS, A*, GBFS, DIJKSTRA")
#         return 1

#     # Parse command-line arguments
#     filename = sys.argv[1]
#     method = sys.argv[2].upper()

#     # Map method strings to search algorithm objects
#     methods = {
#         "BFS": BreadthFirstSearch(),
#         "DFS": DepthFirstSearch(),
#         "A*": AStarSearch(),
#         "GBFS": GreedyBestFirstSearch(),
#         # "DIJKSTRA": DijkstraSearch(),
#     }

#     # Check if the method is valid
#     if method not in methods:
#         print(f"Error: Unknown method '{method}'")
#         print("Available methods: BFS, DFS, A*, GBFS, DIJKSTRA")
#         return 1

#     try:
#         # Reset node counter before each search
#         Node.node_count = 0
        
#         # Load the graph problem from the file
#         problem = GraphProblem.from_file(filename)
        
#         # Use the first goal in the list
#         if problem.goals:
#             problem.current_goal = problem.goals[0]
        
#         # Execute the search algorithm
#         search_algorithm = methods[method]
#         result = search_algorithm.search(problem)
        
#         # Output the results in the required format
#         if result:
#             # First line: filename method
#             print(f"{filename} {method}")
            
#             # Second line: goal number_of_nodes
#             print(f"{problem.current_goal} {Node.node_count}")
            
#             # Third line: path
#             path = [node.state for node in result.path()]
#             print(" ".join(map(str, path)))
            
#             return 0  # Success
#         else:
#             print(f"No solution found for {filename} using {method}")
#             return 1  # No solution
            
#     except FileNotFoundError:
#         print(f"Error: File '{filename}' not found")
#         return 1
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return 1

# if __name__ == "__main__":
#     sys.exit(main())
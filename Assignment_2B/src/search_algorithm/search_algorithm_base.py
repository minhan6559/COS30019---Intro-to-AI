class SearchAlgorithmBase:
    def __init__(self):
        pass

    def search(self, problem):
        """
        Search for a solution to the problem.

        Args:
            problem: A problem object that implements:
                - initial: The initial state
                - goal_test(state): Returns True if state is a goal state
                - get_neighbors(state): Returns a list of neighbors of the state
                - path_cost(c, state1, state2): Returns the cost of moving from state1 to state2
                - h(node): Returns the heuristic value for a node (optional)

        Returns:
            A tuple containing:
            - node: The solution node if found, None otherwise. The node contains:
                - state: The state reached
                - parent: The parent node
                - path_cost: The cost of the path from initial to this node
            - expanded_count: Number of nodes expanded during search
            - created_count: Number of nodes created during search

        Notes:
            - If no solution is found, returns (None, expanded_count, created_count)
            - All search algorithms should reset the Node.nodes_created counter at the start
            - expanded_count tracks nodes that were processed (removed from the frontier)
            - created_count tracks all nodes that were instantiated
        """
        raise NotImplementedError

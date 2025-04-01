from src.graph.graph import Node
from src.search_algorithm.search_algorithm_base import SearchAlgorithmBase
from src.utils.utils import memoize, PriorityQueue

from collections import deque


class DepthFirstSearch(SearchAlgorithmBase):
    def search(self, problem):
        """
        Search the deepest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Does not get trapped by loops.
        If two paths reach a state, only use the first one.
        """
        frontier = [Node(problem.initial)]  # Stack

        explored = set()
        while frontier:
            node = frontier.pop()
            if problem.goal_test(node.state):
                return node
            explored.add(node.state)
            frontier.extend(
                child
                for child in node.expand(problem)
                if child.state not in explored and child not in frontier
            )
        return None


class BreadthFirstSearch(SearchAlgorithmBase):
    def search(self, problem):
        node = Node(problem.initial)
        if problem.goal_test(node.state):
            return node
        frontier = deque([node])
        explored = set()
        while frontier:
            node = frontier.popleft()
            explored.add(node.state)
            for child in node.expand(problem):
                if child.state not in explored and child not in frontier:
                    if problem.goal_test(child.state):
                        return child
                    frontier.append(child)
        return None


class BestFirstSearch(SearchAlgorithmBase):
    def search(self, problem, f, use_memoize=False):
        """Search the nodes with the lowest f scores first.
        You specify the function f(node) that you want to minimize; for example,
        if f is a heuristic estimate to the goal, then we have greedy best
        first search; if f is node.depth then we have breadth-first search.
        There is a subtlety: the line "f = memoize(f, 'f')" means that the f
        values will be cached on the nodes as they are computed. So after doing
        a best first search you can examine the f values of the path returned."""
        if use_memoize:
            f = memoize(f, "f")

        node = Node(problem.initial)
        frontier = PriorityQueue("min", f)
        frontier.append(node)
        explored = set()
        while frontier:
            node = frontier.pop()
            if problem.goal_test(node.state):
                return node
            explored.add(node.state)
            for child in node.expand(problem):
                if child.state not in explored and child not in frontier:
                    frontier.append(child)
                elif child in frontier:
                    if f(child) < frontier[child]:
                        del frontier[child]
                        frontier.append(child)
        return None


class GreedyBestFirstSearch(BestFirstSearch):
    def search(self, problem, use_memoize=False):
        f = problem.h
        return super().search(problem, f, use_memoize)


class AStarSearch(BestFirstSearch):
    def search(self, problem, use_memoize=False):
        h = memoize(problem.h, "h")
        f = lambda n: n.path_cost + h(n)  # f(n) = g(n) + h(n)
        return super().search(problem, f, use_memoize)


class DijkstraSearch(BestFirstSearch):
    def search(self, problem, use_memoize=False):
        f = lambda n: n.path_cost
        return super().search(problem, f, use_memoize)


class IDAStarSearch(SearchAlgorithmBase):
    def search(self, problem):
        """
        Perform Iterative Deepening A* (IDA*) search.
        Uses a depth-limited version of A* with iterative deepening and memoization.
        """
        # Ensure the heuristic is cached using memoization
        h = memoize(problem.h, "h")

        # Recursive search function with pruning
        def search(node, g, bound, explored):
            f = g + h(node)  # f(n) = g(n) + h(n)
            if f > bound:
                return f  # Return the new bound if the limit is exceeded

            if problem.goal_test(node.state):
                return node  # Goal found, return the node

            min_bound = float("inf")

            # Expand the current node
            for child in node.expand(problem):
                if child.state not in explored:
                    explored.add(child.state)
                    temp = search(child, g + child.path_cost, bound, explored)
                    if isinstance(temp, Node):  # If a goal node is found, return it
                        return temp
                    min_bound = min(min_bound, temp)
                    explored.remove(child.state)

            return min_bound

        # Set the initial bound to the heuristic value of the initial node
        initial_node = Node(problem.initial)
        bound = h(initial_node)
        explored = set([problem.initial])  # Start with the initial state in explored set
        result = None

        # Iteratively increase the bound until the solution is found
        while True:
            result = search(initial_node, 0, bound, explored)
            if isinstance(result, Node):
                return result  # Return the actual goal node
            elif result == float("inf"):
                return None  # No solution found within the current bound
            bound = result  # Update the bound for the next iteration


class BidirectionalAStarSearch(SearchAlgorithmBase):
    def search(self, problem, use_memoize=False):
        """
        Perform Bidirectional A* Search.
        This search algorithm runs A* from both the start and goal nodes.
        It stops when the frontiers from both directions meet.
        """
        # Initialize the forward and backward search frontiers
        start_node = Node(problem.initial)
        goal_node = Node(problem.goal)
        
        if problem.goal_test(start_node.state):
            return start_node
        
        # Frontiers for forward and backward searches
        forward_frontier = PriorityQueue("min", lambda n: n.path_cost + problem.h(n))
        forward_frontier.append(start_node)
        
        backward_frontier = PriorityQueue("min", lambda n: n.path_cost + problem.h(n))
        backward_frontier.append(goal_node)
        
        # Explored states for both forward and backward searches
        forward_explored = set()
        backward_explored = set()
        
        while forward_frontier and backward_frontier:
            # Expand the forward frontier
            forward_node = forward_frontier.pop()
            if problem.goal_test(forward_node.state):
                return forward_node
            forward_explored.add(forward_node.state)
            
            for child in forward_node.expand(problem):
                if child.state not in forward_explored:
                    forward_frontier.append(child)
                    
            # Expand the backward frontier
            backward_node = backward_frontier.pop()
            if problem.goal_test(backward_node.state):
                return backward_node
            backward_explored.add(backward_node.state)
            
            for child in backward_node.expand(problem):
                if child.state not in backward_explored:
                    backward_frontier.append(child)
            
            # Check for intersection of forward and backward frontiers
            if forward_frontier[-1].state in backward_explored:
                return forward_frontier[-1]
            
            if backward_frontier[-1].state in forward_explored:
                return backward_frontier[-1]
        
        return None

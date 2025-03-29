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
    def search(self, problem, f):
        """Search the nodes with the lowest f scores first.
        You specify the function f(node) that you want to minimize; for example,
        if f is a heuristic estimate to the goal, then we have greedy best
        first search; if f is node.depth then we have breadth-first search.
        There is a subtlety: the line "f = memoize(f, 'f')" means that the f
        values will be cached on the nodes as they are computed. So after doing
        a best first search you can examine the f values of the path returned."""
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
    def search(self, problem):
        f = problem.h
        return super().search(problem, f)


class AStarSearch(BestFirstSearch):
    def search(self, problem):
        h = memoize(problem.h, "h")
        f = lambda n: n.path_cost + h(n)  # f(n) = g(n) + h(n)
        return super().search(problem, f)


class DijkstraSearch(SearchAlgorithmBase):
    def search(self, problem):
        node = Node(problem.initial)
        frontier = PriorityQueue(
            "min", lambda n: n.path_cost
        )  # Dijkstra uses only g(n)
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
                    if child.path_cost < frontier[child]:
                        del frontier[child]
                        frontier.append(child)
        return None


class IDAStarSearch(SearchAlgorithmBase):
    def search(self, problem):
        """
        Perform Iterative Deepening A* (IDA*) search.
        Uses a depth-limited version of A* with iterative deepening.
        """
        def search(node, g, bound):
            # f(n) = g(n) + h(n)
            f = g + problem.h(node)
            if f > bound:
                return f  # Return the new bound if the limit is exceeded
            if problem.goal_test(node.state):
                return node  # Goal found, return the node
            min_bound = float('inf')

            for child in node.expand(problem):
                if child.state not in explored:
                    explored.add(child.state)
                    temp = search(child, g + child.path_cost, bound)
                    if isinstance(temp, Node):  # If a goal node is found, return it
                        return temp
                    min_bound = min(min_bound, temp)
                    explored.remove(child.state)

            return min_bound

        bound = problem.h(Node(problem.initial))  # Set the initial bound to h(n)
        explored = set([problem.initial])  # Start with the initial state in explored set
        result = None

        while True:
            result = search(Node(problem.initial), 0, bound)
            if isinstance(result, Node):
                return result  # Return the actual goal node instead of the path
            elif result == float('inf'):
                return None
            bound = result  # Update the bound for the next iteration

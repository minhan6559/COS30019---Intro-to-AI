from graph import Node
from search_algorithm_base import SearchAlgorithmBase
from collections import deque
from utils import *


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


class GreedyBestFirstSearch(SearchAlgorithmBase):
    def search(self, problem):
        node = Node(problem.initial)
        frontier = PriorityQueue("min", problem.h)  # Use the heuristic to prioritize
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
        return None



class AStarSearch(SearchAlgorithmBase):
    def search(self, problem):
        h = memoize(problem.h, "h")
        node = Node(problem.initial)
        frontier = PriorityQueue("min", lambda n: n.path_cost + h(n))  # A* f(n) = g(n) + h(n)
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
                    if (child.path_cost + h(child)) < frontier[child]:
                        del frontier[child]
                        frontier.append(child)
        return None


class DijkstraSearch(SearchAlgorithmBase):
    def search(self, problem):
        node = Node(problem.initial)
        frontier = PriorityQueue("min", lambda n: n.path_cost)  # Dijkstra uses only g(n)
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
        def search(node, g, bound):
            # Calculate f(n) = g(n) + h(n)
            f = g + problem.h(node)
            if f > bound:  # If the cost exceeds the current bound, return the value of f
                return f
            if problem.goal_test(node.state):  # If goal is found, return the path
                return node
            min_bound = float('inf')
            
            # Expand all children of the current node
            for child in node.expand(problem):
                if child.state not in explored:  # Avoid cycles
                    explored.add(child.state)
                    temp = search(child, g + child.path_cost, bound)
                    if isinstance(temp, Node):  # Goal found, return the path
                        return temp
                    min_bound = min(min_bound, temp)  # Update the minimum bound
                    explored.remove(child.state)
            
            return min_bound  # Return the minimum bound for this path

        bound = problem.h(problem.initial)  # Initial bound is just the heuristic of the start node
        explored = set()
        node = Node(problem.initial)

        while True:
            result = search(node, 0, bound)  # Start search with cost 0 and current bound
            if isinstance(result, Node):  # Goal found, return the path
                return result
            if result == float('inf'):
                return None
            bound = result  # Increase the bound for the next iteration

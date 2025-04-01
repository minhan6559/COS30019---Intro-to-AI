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
        h = problem.h
        f = lambda n: n.path_cost + h(n)  # f(n) = g(n) + h(n)
        return super().search(problem, f)


class UniformCostSearch(BestFirstSearch):
    def search(self, problem):
        f = lambda n: n.path_cost
        return super().search(problem, f)


class BULBSearch(SearchAlgorithmBase):
    def __init__(self, beam_width=10, max_discrepancies=10):
        self.beam_width = beam_width
        self.max_discrepancies = max_discrepancies

    def search(self, problem):
        h = problem.h

        class DiscrepancyNode:
            def __init__(self, node, discrepancies=0):
                self.node = node
                self.discrepancies = discrepancies
                self.f_value = node.path_cost + h(node)

            def __lt__(self, other):
                if self.discrepancies != other.discrepancies:
                    return self.discrepancies < other.discrepancies
                return self.f_value < other.f_value

        # Start with the initial node
        initial_node = DiscrepancyNode(Node(problem.initial))

        # Use PriorityQueue instead of a list
        queue = PriorityQueue("min", lambda n: (n.discrepancies, n.f_value))
        queue.append(initial_node)
        visited = set()

        while queue:
            current = queue.pop()

            if current.node.state in visited:
                continue
            visited.add(current.node.state)

            if problem.goal_test(current.node.state):
                return current.node

            # Generate successors
            successors = [
                DiscrepancyNode(child, current.discrepancies)
                for child in current.node.expand(problem)
            ]

            if not successors:
                continue

            # Sort by f-value for beam selection
            successors.sort(key=lambda n: n.f_value)

            # Apply beam search
            beam = successors[: min(self.beam_width, len(successors))]
            pruned = successors[min(self.beam_width, len(successors)) :]

            # Add beam nodes to the queue
            for node in beam:
                queue.append(node)

            # Add pruned nodes with increased discrepancy if within limit
            for node in pruned:
                if node.discrepancies < self.max_discrepancies:
                    backtrack_node = DiscrepancyNode(node.node, node.discrepancies + 1)
                    queue.append(backtrack_node)

        return None

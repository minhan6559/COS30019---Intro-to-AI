from src.graph.node import Node, DiscrepancyNode
from src.search_algorithm.search_algorithm_base import SearchAlgorithmBase
from src.utils.utils import PriorityQueue

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
        # Reset the node creation counter
        Node.reset_counter()

        initial_node = Node(problem.initial)

        stack = [initial_node]
        count_expanded = 0  # Initialize the count of nodes expanded
        visited = set()

        while stack:
            node = stack.pop()
            count_expanded += 1  # Increment the count for each node expanded

            if problem.goal_test(node.state):
                return node, count_expanded, Node.nodes_created

            visited.add(node.state)

            children = node.expand(problem, reverse=True)

            stack.extend(child for child in children if child.state not in visited)

        return None, count_expanded, Node.nodes_created


class BreadthFirstSearch(SearchAlgorithmBase):
    def search(self, problem):
        Node.reset_counter()  # Reset the node creation counter

        node = Node(problem.initial)

        queue = deque([node])
        count_expanded = 0  # Initialize the count of nodes expanded
        visited = set()

        while queue:

            node = queue.popleft()
            count_expanded += 1

            if problem.goal_test(node.state):
                return node, count_expanded, Node.nodes_created

            visited.add(node.state)
            children = node.expand(problem)

            for child in children:
                if child.state not in visited and child not in queue:
                    if problem.goal_test(child.state):
                        return child, count_expanded, Node.nodes_created

                    queue.append(child)

        return None, count_expanded, Node.nodes_created


class BestFirstSearch(SearchAlgorithmBase):
    def search(self, problem, f):
        Node.reset_counter()  # Reset the node creation counter

        node = Node(problem.initial)

        priority_queue = PriorityQueue("min", f)
        priority_queue.append(node)
        visited = set()
        count_expanded = 0

        while priority_queue:
            node = priority_queue.pop()
            count_expanded += 1

            if problem.goal_test(node.state):
                return node, count_expanded, Node.nodes_created

            visited.add(node.state)

            for child in node.expand(problem):
                if child.state not in visited and child not in priority_queue:
                    priority_queue.append(child)
                elif child in priority_queue:
                    if f(child) < priority_queue[child]:
                        del priority_queue[child]
                        priority_queue.append(child)

        return None, count_expanded, Node.nodes_created


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


# Beam search using Limited Discrepancy Backtracking Search (BULB)
class BULBSearch(SearchAlgorithmBase):
    def __init__(self, beam_width=50, max_discrepancies=8):
        self.beam_width = beam_width
        self.max_discrepancies = max_discrepancies

    def search(self, problem):
        Node.reset_counter()  # Reset the node creation counter

        f_func = lambda n: n.path_cost + problem.h(n)

        # Create initial node and wrap it
        initial_node = Node(problem.initial)
        initial_discrepancy_node = DiscrepancyNode(initial_node, f_func)

        # Create a node cache to avoid recreating nodes for the same state
        node_cache = {problem.initial: initial_node}

        priority_queue = PriorityQueue("min", lambda n: (n.discrepancies, n.f_value))
        priority_queue.append(initial_discrepancy_node)
        visited = set()
        count_expanded = 0

        while priority_queue:
            current = priority_queue.pop()
            count_expanded += 1

            if current.node.state in visited:
                continue

            if problem.goal_test(current.node.state):
                return current.node, count_expanded, Node.nodes_created

            visited.add(current.node.state)

            # Get neighbors without creating nodes yet
            neighbors = problem.get_neighbors(current.node.state)
            successors = []

            for next_state in neighbors:
                # Create or reuse a node for this state
                if next_state in node_cache:
                    # Node already exists in our cache
                    child_node = node_cache[next_state]

                    # Update path cost if we found a better path
                    new_cost = problem.path_cost(
                        current.node.path_cost, current.node.state, next_state
                    )
                    if new_cost < child_node.path_cost:
                        child_node.path_cost = new_cost
                        child_node.parent = current.node
                else:
                    # Create a new node
                    child_node = current.node.child_node(problem, next_state)
                    node_cache[next_state] = child_node

                # Create a discrepancy node with the node
                successors.append(
                    DiscrepancyNode(child_node, f_func, current.discrepancies)
                )

            if not successors:
                continue

            # Sort by both f-value and state for consistent ordering
            successors.sort(key=lambda n: (n.f_value, n.node.state))

            # Apply beam search
            beam = successors[: min(self.beam_width, len(successors))]
            pruned = successors[min(self.beam_width, len(successors)) :]

            # Add beam nodes to the priority_queue
            for node in beam:
                priority_queue.append(node)

            # Add pruned nodes with increased discrepancy if within limit
            for node in pruned:
                if node.discrepancies < self.max_discrepancies:
                    backtrack_node = DiscrepancyNode(
                        node.node, f_func, node.discrepancies + 1
                    )
                    priority_queue.append(backtrack_node)

        return None, count_expanded, Node.nodes_created

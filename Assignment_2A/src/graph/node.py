class Node:
    # Class variable to count the total number of nodes created
    nodes_created = 0

    def __init__(self, state, parent=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost

        # Increment the nodes created counter
        Node.nodes_created += 1

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem, should_sort=True, reverse=False):
        """
        List the nodes reachable in one step from this node.
        The result is a list of nodes, not states.
        The nodes are sorted in order based on their state.
        """
        children = [
            self.child_node(problem, neighbor)
            for neighbor in problem.get_neighbors(self.state)
        ]

        if should_sort:
            children.sort(
                key=lambda n: n.state, reverse=reverse
            )  # Sort children by state in order

        return children

    def child_node(self, problem, next_state):
        next_node = Node(
            next_state,
            self,
            problem.path_cost(self.path_cost, self.state, next_state),
        )
        return next_node

    def path_nodes(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def path_states(self):
        """Return the list of states from the root to this node."""
        return [node.state for node in self.path_nodes()]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

    @classmethod
    def reset_counter(cls):
        """Reset the nodes_created counter to zero."""
        cls.nodes_created = 0


class DiscrepancyNode:
    def __init__(self, node, f_func, discrepancies=0):
        self.node = node
        self.discrepancies = discrepancies
        self.f_value = f_func(node)

    def __lt__(self, other):
        if self.discrepancies != other.discrepancies:
            return self.discrepancies < other.discrepancies
        if self.f_value != other.f_value:
            return self.f_value < other.f_value
        return self.node.state < other.node.state  # Add this for proper ordering

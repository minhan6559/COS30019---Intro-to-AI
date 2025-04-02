class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

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
        """[Figure 3.10]"""
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

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


class DiscrepancyNode:
    def __init__(self, node, h_func, discrepancies=0):
        self.node = node
        self.discrepancies = discrepancies
        self.f_value = node.path_cost + h_func(node)

    def __lt__(self, other):
        if self.discrepancies != other.discrepancies:
            return self.discrepancies < other.discrepancies
        if self.f_value != other.f_value:
            return self.f_value < other.f_value
        return self.node.state < other.node.state  # Add this for proper ordering

from src.problem.problem_base import ProblemBase
from src.graph.graph import Graph
from src.utils.utils import distance

import numpy as np
import random


class GraphProblem(ProblemBase):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph, locations):
        super().__init__(initial, goal)
        self.graph = graph
        self.locations = locations

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, "locations", None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf


def generate_random_graph_problem(
    num_nodes=10,
    min_edges_per_node=2,
    max_edges_per_node=4,
    grid_size=100,
    num_destinations=1,
    curvature=lambda: random.uniform(1.1, 1.5),
):
    """
    Generate a random graph problem for testing search algorithms.

    Args:
        num_nodes: Number of nodes in the graph
        min_edges_per_node: Minimum outgoing edges per node
        max_edges_per_node: Maximum outgoing edges per node
        grid_size: Size of the coordinate grid (for node locations)
        num_destinations: Number of destination nodes

    Returns:
        problem: A GraphProblem instance
    """
    # Generate node IDs and random locations
    nodes = list(range(1, num_nodes + 1))
    locations = {
        node: (random.randint(0, grid_size), random.randint(0, grid_size))
        for node in nodes
    }

    # Create empty graph
    graph_dict = {node: {} for node in nodes}

    # Add random edges
    for node in nodes:
        # Determine number of outgoing edges for this node
        num_edges = random.randint(
            min_edges_per_node, min(max_edges_per_node, num_nodes - 1)
        )

        # Get potential targets (all nodes except self)
        potential_targets = [n for n in nodes if n != node]

        potential_targets.sort(key=lambda x: distance(locations[node], locations[x]))

        # Select targets
        targets = potential_targets[: min(num_edges, len(potential_targets))]

        # Add edges with random weights
        for target in targets:
            weight = distance(locations[node], locations[target]) * curvature()
            graph_dict[node][target] = weight

    # Create the graph
    graph = Graph(graph_dict)

    # Select random origin and destinations
    origin = random.choice(nodes)
    remaining_nodes = [n for n in nodes if n != origin]
    destinations = random.sample(
        remaining_nodes, min(num_destinations, len(remaining_nodes))
    )

    # Create the problem
    problem = GraphProblem(origin, destinations, graph, locations)

    return problem

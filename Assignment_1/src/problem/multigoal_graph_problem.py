from src.problem.problem_base import ProblemBase
from src.graph.graph import Graph
from src.utils.utils import distance

import random
import math
from collections import deque


class MultigoalGraphProblem(ProblemBase):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goals, graph, locations):
        """
        Initialize the graph problem with a starting node and goal(s).

        Args:
            initial: The starting node
            goals: Either a single goal node or a list of possible goal nodes
            graph: The graph to search
            locations: A dictionary mapping nodes to (x, y) coordinates
        """
        # Convert single goal to a list for consistency
        self.goals = [goals] if not isinstance(goals, list) else goals
        super().__init__(initial, self.goals)
        self.graph = graph
        self.locations = locations

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or math.inf)

    def goal_test(self, state):
        """
        Return True if the state matches one of the goals.
        """
        return state in self.goals

    def h(self, node):
        """
        Heuristic function to estimate the cost from the current node to the nearest goal.
        """

        if not self.locations or not self.goals:
            return math.inf

        # Handle different node types (Node objects, strings, integers, etc.)
        node_state = node.state if hasattr(node, "state") else node

        # Ensure current node and all goals exist in the locations
        if node_state not in self.locations or not all(
            goal in self.locations for goal in self.goals
        ):
            return math.inf

        # calculate the min distance between the node to all goals
        return int(
            min(
                distance(self.locations[node_state], self.locations[goal])
                for goal in self.goals
            )
        )

    def __repr__(self):
        """Return a string representation of the GraphProblem."""
        nodes_count = len(self.graph.nodes()) if hasattr(self.graph, "nodes") else 0
        edges_count = (
            sum(len(neighbors) for neighbors in self.graph.graph_dict.values())
            if hasattr(self.graph, "graph_dict")
            else 0
        )
        has_locations = bool(self.locations)

        return (
            f"GraphProblem(initial={self.initial}, "
            f"goals={self.goals}, "
            f"nodes={nodes_count}, "
            f"edges={edges_count}, "
            f"has_locations={has_locations})"
        )

    @classmethod
    def has_path(cls, graph_dict, start, goal):
        """
        Check if there's a path from start to goal in the graph.

        Args:
            graph_dict: Dictionary representation of the graph
            start: Starting node
            goal: Goal node

        Returns:
            bool: True if a path exists, False otherwise
        """

        visited = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node == goal:
                return True

            if node not in visited:
                visited.add(node)
                queue.extend([n for n in graph_dict[node] if n not in visited])

        return False

    @classmethod
    def create_connectivity_path(
        cls, graph_dict, start, goal, locations, nodes, curvature
    ):
        """
        Create a path from start to goal, potentially using intermediate nodes.

        Args:
            graph_dict: Dictionary representation of the graph
            start: Starting node
            goal: Goal node
            locations: Dictionary mapping nodes to coordinates
            nodes: List of all nodes in the graph
            curvature: Function that returns a factor to multiply distances by
        """
        # First check if there's already a path
        if cls.has_path(graph_dict, start, goal):
            return

        # Find all nodes reachable from start
        reachable = set()
        queue = deque([start])
        visited = {start}

        while queue:
            node = queue.popleft()
            reachable.add(node)
            for next_node in graph_dict[node]:
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append(next_node)

        # If goal is already reachable, we're done
        if goal in reachable:
            return

        # Try to connect reachable nodes to unreachable nodes including goal
        if len(reachable) < len(nodes):
            # Pick the closest pair of nodes between reachable and unreachable sets
            best_src, best_dest = None, None
            best_dist = float("inf")

            for src in reachable:
                for dest in nodes:
                    if dest not in reachable:
                        dist = distance(locations[src], locations[dest])
                        if dist < best_dist:
                            best_dist = dist
                            best_src, best_dest = src, dest

            if best_src and best_dest:
                # Add an edge to connect the two partitions
                weight = best_dist * curvature()
                graph_dict[best_src][best_dest] = weight

                # Check if this creates a path to the goal
                if not cls.has_path(graph_dict, start, goal):
                    # Recursively try again
                    cls.create_connectivity_path(
                        graph_dict, start, goal, locations, nodes, curvature
                    )
                return

        # If all else fails, create a direct path
        weight = distance(locations[start], locations[goal]) * curvature()
        graph_dict[start][goal] = weight

    @classmethod
    def random(
        cls,
        num_nodes=10,
        min_edges_per_node=2,
        max_edges_per_node=4,
        grid_size=100,
        num_destinations=1,
        curvature=lambda: random.uniform(0.8, 2.5),
        max_distance_factor=2.0,
        ensure_connectivity=True,
    ):
        """
        Generate a random graph problem for testing search algorithms.

        Args:
            num_nodes: Number of nodes in the graph
            min_edges_per_node: Minimum outgoing edges per node
            max_edges_per_node: Maximum outgoing edges per node
            grid_size: Size of the coordinate grid (for node locations)
            num_destinations: Number of destination nodes
            curvature: Function that returns a factor to multiply distances by
            max_distance_factor: Maximum distance for connections, as a multiple of the average neighbor distance
            ensure_connectivity: If True, ensures the origin can reach all destinations

        Returns:
            problem: A GraphProblem instance
        """
        # Generate node IDs and random locations
        nodes, locations = cls._generate_random_nodes(num_nodes, grid_size)

        # Select random origin and destinations
        origin, destinations = cls._select_origin_and_destinations(
            nodes, num_destinations
        )

        # Calculate maximum connection distance
        max_connection_distance = cls._calculate_max_connection_distance(
            nodes, locations, max_distance_factor, grid_size
        )

        # Create graph with random edges
        graph_dict = cls._generate_random_edges(
            nodes,
            locations,
            min_edges_per_node,
            max_edges_per_node,
            max_connection_distance,
            curvature,
        )

        # Ensure connectivity if requested
        if ensure_connectivity:
            cls._ensure_graph_connectivity(
                graph_dict, origin, destinations, locations, nodes, curvature
            )

        # Create the graph and problem
        graph = Graph(graph_dict)
        return cls(origin, destinations, graph, locations)

    @classmethod
    def _generate_random_nodes(cls, num_nodes, grid_size):
        """
        Generate random nodes and their locations.

        Args:
            num_nodes: Number of nodes to generate
            grid_size: Size of the coordinate grid

        Returns:
            tuple: (nodes list, locations dictionary)
        """
        nodes = list(range(1, num_nodes + 1))
        locations = {
            node: (random.randint(0, grid_size), random.randint(0, grid_size))
            for node in nodes
        }
        return nodes, locations

    @classmethod
    def _select_origin_and_destinations(cls, nodes, num_destinations):
        """
        Select random origin and destination nodes.

        Args:
            nodes: List of all nodes
            num_destinations: Number of destinations to select

        Returns:
            tuple: (origin node, list of destination nodes)
        """
        origin = random.choice(nodes)
        remaining_nodes = [n for n in nodes if n != origin]
        destinations = random.sample(
            remaining_nodes, min(num_destinations, len(remaining_nodes))
        )
        return origin, destinations

    @classmethod
    def _calculate_max_connection_distance(
        cls, nodes, locations, max_distance_factor, grid_size
    ):
        """
        Calculate maximum allowed connection distance based on average nearest neighbor.

        Args:
            nodes: List of all nodes
            locations: Dictionary mapping nodes to coordinates
            max_distance_factor: Factor to multiply average distance by
            grid_size: Size of the coordinate grid

        Returns:
            float: Maximum connection distance
        """
        all_distances = []
        for node in nodes:
            node_distances = [
                distance(locations[node], locations[other])
                for other in nodes
                if other != node
            ]
            node_distances.sort()
            if node_distances:  # Ensure there are neighbors
                all_distances.append(
                    node_distances[0]
                )  # Add distance to closest neighbor

        avg_neighbor_distance = (
            sum(all_distances) / len(all_distances) if all_distances else grid_size / 4
        )
        return avg_neighbor_distance * max_distance_factor

    @classmethod
    def _generate_random_edges(
        cls,
        nodes,
        locations,
        min_edges_per_node,
        max_edges_per_node,
        max_connection_distance,
        curvature,
    ):
        """
        Generate random edges for the graph.

        Args:
            nodes: List of all nodes
            locations: Dictionary mapping nodes to coordinates
            min_edges_per_node: Minimum outgoing edges per node
            max_edges_per_node: Maximum outgoing edges per node
            max_connection_distance: Maximum allowed connection distance
            curvature: Function that returns a factor to multiply distances by

        Returns:
            dict: Graph dictionary with random edges
        """
        graph_dict = {node: {} for node in nodes}

        for node in nodes:
            # Get potential targets within max distance
            potential_targets = [
                n
                for n in nodes
                if n != node
                and distance(locations[node], locations[n]) <= max_connection_distance
            ]

            # If not enough targets within range, relax the constraint
            if len(potential_targets) < min_edges_per_node:
                potential_targets = [n for n in nodes if n != node]

            # Sort by distance
            potential_targets.sort(
                key=lambda x: distance(locations[node], locations[x])
            )

            # Determine number of edges for this node
            num_edges = min(
                random.randint(min_edges_per_node, max_edges_per_node),
                len(potential_targets),
            )

            # Select targets and add edges
            targets = potential_targets[:num_edges]
            for target in targets:
                weight = distance(locations[node], locations[target]) * curvature()
                graph_dict[node][target] = weight

        return graph_dict

    @classmethod
    def _ensure_graph_connectivity(
        cls, graph_dict, origin, destinations, locations, nodes, curvature
    ):
        """
        Ensure the graph has paths from origin to all destinations.

        Args:
            graph_dict: Dictionary representation of the graph
            origin: Origin node
            destinations: List of destination nodes
            locations: Dictionary mapping nodes to coordinates
            nodes: List of all nodes
            curvature: Function that returns a factor to multiply distances by
        """
        for dest in destinations:
            cls.create_connectivity_path(
                graph_dict, origin, dest, locations, nodes, curvature
            )

    @classmethod
    def from_file(cls, filename):
        """
        Create a GraphProblem instance from a file.

        Args:
            filename: Path to the graph file

        Returns:
            GraphProblem: An instance created from the file data
        """
        from src.parser.graph_parser import GraphParser

        parser = GraphParser()
        graph, origin, destinations, locations = parser.parse_file(
            filename
        ).get_problem_components()

        # Create and return the problem instance
        return cls(origin, destinations, graph, locations)

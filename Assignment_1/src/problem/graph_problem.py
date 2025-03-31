from src.problem.problem_base import ProblemBase
from src.graph.graph import Graph
from src.utils.utils import distance

import numpy as np
import random


class GraphProblem(ProblemBase):
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
        # Track the current active goal
        self.current_goal = self.goals[0] if self.goals else None

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

    def goal_test(self, state):
        """
        Return True if the state matches the current active goal.
        This overrides the default method to test only against the current goal.
        """
        return state == self.current_goal

    def h(self, node):
        """h function is straight-line distance from node's state to current goal."""
        if not self.locations or not self.current_goal:
            return np.inf

        # Handle different node types (Node objects, strings, integers, etc.)
        node_state = node.state if hasattr(node, "state") else node

        # Ensure both locations exist
        if node_state not in self.locations or self.current_goal not in self.locations:
            return np.inf

        return int(
            distance(self.locations[node_state], self.locations[self.current_goal])
        )

    def set_current_goal(self, goal):
        """
        Change the current active goal.

        Args:
            goal: The new goal to set as current destination

        Returns:
            bool: True if goal was changed successfully, False otherwise
        """
        if goal in self.goals:
            self.current_goal = goal
            return True
        return False

    def next_goal(self):
        """
        Switch to the next goal in the list of goals.

        Returns:
            The new current goal, or None if there are no goals
        """
        if not self.goals:
            return None

        current_index = (
            self.goals.index(self.current_goal)
            if self.current_goal in self.goals
            else -1
        )
        next_index = (current_index + 1) % len(self.goals)
        self.current_goal = self.goals[next_index]
        return self.current_goal

    def get_remaining_goals(self):
        """
        Get list of goals that haven't been reached yet.

        Returns:
            list: All goals in the goals list excluding the current one
        """
        return [g for g in self.goals if g != self.current_goal]

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
            f"current_goal={self.current_goal}, "
            f"nodes={nodes_count}, "
            f"edges={edges_count}, "
            f"has_locations={has_locations})"
        )

    @classmethod
    def random(
        cls,
        num_nodes=10,
        min_edges_per_node=2,
        max_edges_per_node=4,
        grid_size=100,
        num_destinations=1,
        curvature=lambda: random.uniform(1.1, 1.5),
        max_distance_factor=2.0,
        ensure_connectivity=True,  # New parameter
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
        nodes = list(range(1, num_nodes + 1))
        locations = {
            node: (random.randint(0, grid_size), random.randint(0, grid_size))
            for node in nodes
        }

        # Select random origin and destinations before building the graph
        origin = random.choice(nodes)
        remaining_nodes = [n for n in nodes if n != origin]
        destinations = random.sample(
            remaining_nodes, min(num_destinations, len(remaining_nodes))
        )

        # Calculate average nearest neighbor distance
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
        max_connection_distance = avg_neighbor_distance * max_distance_factor

        # Create empty graph
        graph_dict = {node: {} for node in nodes}

        # Add random edges
        for node in nodes:
            # Get potential targets (all nodes except self within max distance)
            potential_targets = [
                n
                for n in nodes
                if n != node
                and distance(locations[node], locations[n]) <= max_connection_distance
            ]

            # If not enough targets within range, relax the constraint to ensure min_edges_per_node
            if len(potential_targets) < min_edges_per_node:
                potential_targets = [n for n in nodes if n != node]

            potential_targets.sort(
                key=lambda x: distance(locations[node], locations[x])
            )

            # Determine number of outgoing edges for this node (limited by available targets)
            num_edges = min(
                random.randint(min_edges_per_node, max_edges_per_node),
                len(potential_targets),
            )

            # Select targets
            targets = potential_targets[:num_edges]

            # Add edges with random weights
            for target in targets:
                weight = distance(locations[node], locations[target]) * curvature()
                graph_dict[node][target] = weight

        # If ensure_connectivity is True, ensure there are paths from origin to destinations
        if ensure_connectivity:
            # Define helper functions for path checking and creation
            def has_path(graph_dict, start, goal):
                """Check if there's a path from start to goal in the graph."""
                visited = set()
                queue = [start]

                while queue:
                    node = queue.pop(0)
                    if node == goal:
                        return True

                    if node not in visited:
                        visited.add(node)
                        queue.extend([n for n in graph_dict[node] if n not in visited])

                return False

            def create_connectivity_path(
                graph_dict, start, goal, locations, nodes, curvature
            ):
                """Create a path from start to goal, potentially using intermediate nodes."""
                # First check if there's already a path
                if has_path(graph_dict, start, goal):
                    return

                # Find all nodes reachable from start
                reachable = set()
                queue = [start]
                visited = {start}

                while queue:
                    node = queue.pop(0)
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
                        if not has_path(graph_dict, start, goal):
                            # Recursively try again
                            create_connectivity_path(
                                graph_dict, start, goal, locations, nodes, curvature
                            )
                        return

                # If all else fails, create a direct path
                weight = distance(locations[start], locations[goal]) * curvature()
                graph_dict[start][goal] = weight

            # For each destination, ensure there's a path from origin
            for dest in destinations:
                create_connectivity_path(
                    graph_dict, origin, dest, locations, nodes, curvature
                )

        # Create the graph
        graph = Graph(graph_dict)

        # Create the problem
        return cls(origin, destinations, graph, locations)

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

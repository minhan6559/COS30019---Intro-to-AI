from src.problem.problem_base import ProblemBase
from src.graph.graph import Graph

import random
import math
from collections import deque
from haversine import haversine, Unit


class VicRoadsGraphProblem(ProblemBase):
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

    def get_neighbors(self, current_state):
        return list(self.graph.get(current_state).keys())

    def path_cost(self, cost_so_far, current_state, next_state):
        return cost_so_far + (self.graph.get(current_state, next_state) or math.inf)

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

        # Coordinates of the current node
        node_coords = self.locations[node_state]

        # calculate the min travel times in minutes between the node to all goals
        max_speed = 60  # km/h
        return min(
            haversine(node_coords, self.locations[goal], unit=Unit.KILOMETERS)
            / max_speed
            * 60
            for goal in self.goals
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

    @classmethod
    def to_file(cls, problem, filepath):
        """
        Save a graph problem to a file in the standard format.

        Args:
            problem: MultigoalGraphProblem instance to save
            filepath: Path to save the file

        Returns:
            str: The filepath where the problem was saved
        """
        # Extract necessary components
        graph_dict = (
            problem.graph.graph_dict if hasattr(problem.graph, "graph_dict") else {}
        )
        locations = problem.locations
        initial = problem.initial
        goals = problem.goals

        # Ensure all nodes in locations appear in the graph dictionary
        for node in locations:
            if node not in graph_dict:
                graph_dict[node] = {}

        with open(filepath, "w") as f:
            # Write nodes section
            f.write("Nodes:\n")
            for node in sorted(locations.keys()):
                x, y = locations[node]
                f.write(f"{node}: ({x},{y})\n")

            # Write edges section
            f.write("Edges:\n")
            for node in sorted(graph_dict.keys()):
                for neighbor, weight in sorted(graph_dict[node].items()):
                    # Format weight: keep as float with 2 decimal place if it has fractional part
                    if isinstance(weight, float) and weight.is_integer():
                        weight_str = int(weight)
                    else:
                        weight_str = round(float(weight), 2)
                    f.write(f"({node},{neighbor}): {weight_str}\n")

            # Write origin section
            f.write("Origin:\n")
            f.write(f"{initial}\n")

            # Write destinations section
            f.write("Destinations:\n")
            f.write("; ".join(str(goal) for goal in goals))

        return filepath

from src.problem.problem_base import ProblemBase
from src.graph.graph import Graph

import random
import math
from datetime import datetime, timedelta
from collections import deque
from haversine import haversine, Unit


class VicRoadsGraphProblem(ProblemBase):
    """The problem of searching a graph from one node to another with time-dependent costs."""

    def __init__(
        self,
        initial,
        goals,
        graph,
        locations,
        initial_timestamp=None,
        traffic_volume_lookups=None,
        connections=None,
        prediction_model=None,
    ):
        """
        Initialize the graph problem with a starting node and goal(s).

        Args:
            initial: The starting node
            goals: Either a single goal node or a list of possible goal nodes
            graph: The graph to search
            locations: A dictionary mapping nodes to (x, y) coordinates
            initial_timestamp: The starting timestamp for the route
            traffic_volume_lookups: Dictionary structure for looking up traffic volumes by time
            connections: List of connections with distance and approach_location
            prediction_model: The traffic prediction model to use (LSTM, GRU, CNN_LSTM)
        """
        # Convert single goal to a list for consistency
        self.goals = [goals] if not isinstance(goals, list) else goals
        super().__init__(initial, self.goals)
        self.graph = graph
        self.locations = locations

        # New attributes for time-dependent costs
        self.initial_timestamp = initial_timestamp
        self.traffic_volume_lookups = traffic_volume_lookups
        self.connections = connections
        self.prediction_model = prediction_model

        # Cache connection data for faster lookup
        self.connection_lookup = {}
        if connections:
            for conn in connections:
                from_id = int(conn["from_id"])
                to_id = int(conn["to_id"])
                key = (from_id, to_id)
                self.connection_lookup[key] = conn

    def get_neighbors(self, current_state):
        return list(self.graph.get(current_state).keys())

    def path_cost(self, cost_so_far, current_state, next_state):
        """
        Calculate the path cost from current_state to next_state using time-dependent traffic volumes.

        Args:
            cost_so_far: The cost up to current_state (in minutes)
            current_state: Current node ID
            next_state: Next node ID

        Returns:
            Total path cost from initial state to next_state (in minutes)
        """
        # If no time-dependent data is provided, fall back to static costs
        if (
            not self.initial_timestamp
            or not self.traffic_volume_lookups
            or not self.connections
            or not self.prediction_model
        ):
            return cost_so_far + (self.graph.get(current_state, next_state) or math.inf)

        # Get connection details
        conn_key = (int(current_state), int(next_state))
        connection = self.connection_lookup.get(conn_key)

        if not connection:
            return math.inf  # No connection found

        # Get the distance and location
        distance = connection["distance"]  # in km
        location = connection.get("approach_location", "").strip()

        # Calculate the actual timestamp for this leg of the journey
        if isinstance(self.initial_timestamp, str):
            initial_dt = datetime.strptime(self.initial_timestamp, "%Y-%m-%d %H:%M")
        else:
            initial_dt = self.initial_timestamp

        # Add the accumulated cost_so_far (in minutes) to the initial timestamp
        current_time = initial_dt + timedelta(minutes=cost_so_far)

        # Format current time for lookup
        date_str = current_time.date().strftime("%Y-%m-%d")

        # Calculate interval_id (0-95) based on time (15-minute intervals, 4 per hour)
        hour = current_time.hour
        minute = current_time.minute
        interval_id = (hour * 4) + (minute // 15)

        # Get traffic volume for the current time
        traffic_volume = self._get_traffic_volume_with_fallback(
            location, date_str, interval_id
        )

        # Calculate travel time based on distance and traffic volume
        travel_time = self._calculate_travel_time(distance, traffic_volume)

        # Return the total cost up to next_state
        return cost_so_far + travel_time

    def _get_traffic_volume_with_fallback(self, location, date_str, interval_id):
        """
        Get traffic volume with intelligent fallback mechanisms for missing data.

        Strategies (in order of preference):
        1. Use exact time data if available
        2. Use same time data from same day of week in available data
        3. Use historical average for same day of week and time
        4. Use historical average for same time of day
        5. Use default value based on time of day

        Args:
            location: The location ID
            date_str: Date string in format YYYY-MM-DD
            interval_id: Time interval ID (0-95, each representing 15 minutes)

        Returns:
            Traffic volume (non-negative integer)
        """
        model_dict = self.traffic_volume_lookups.get(self.prediction_model, {})
        location_dict = model_dict.get(location, {})

        # Strategy 1: Try to get exact data for the date and time
        date_dict = location_dict.get(date_str, {})
        traffic_volume = date_dict.get(interval_id, None)
        if traffic_volume is not None:
            return max(traffic_volume, 0)

        # At this point, we need to use fallback strategies

        try:
            # Parse the date string to get day of week
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            day_of_week = date_obj.weekday()  # 0=Monday, 6=Sunday

            # Strategy 2: Find same day of week in available data
            same_day_volumes = []
            for available_date in location_dict:
                try:
                    available_date_obj = datetime.strptime(
                        available_date, "%Y-%m-%d"
                    ).date()
                    if available_date_obj.weekday() == day_of_week:
                        vol = location_dict[available_date].get(interval_id, None)
                        if vol is not None:
                            same_day_volumes.append(vol)
                except ValueError:
                    continue

            if same_day_volumes:
                return max(int(sum(same_day_volumes) / len(same_day_volumes)), 0)

            # Strategy 3: Get average for this time interval across all dates
            all_volumes_for_interval = []
            for date_data in location_dict.values():
                vol = date_data.get(interval_id, None)
                if vol is not None:
                    all_volumes_for_interval.append(vol)

            if all_volumes_for_interval:
                return max(
                    int(sum(all_volumes_for_interval) / len(all_volumes_for_interval)),
                    0,
                )

            # Strategy 4: Default based on time of day
            # Morning peak (7-9 AM)
            if 28 <= interval_id <= 35:  # 7:00-9:00 AM
                return 25  # Higher traffic
            # Evening peak (4-6 PM)
            elif 64 <= interval_id <= 71:  # 16:00-18:00
                return 30  # Highest traffic
            # Mid-day (9 AM - 4 PM)
            elif 36 <= interval_id <= 63:  # 9:00-16:00
                return 15  # Medium traffic
            # Night (6 PM - 7 AM)
            else:
                return 5  # Low traffic

        except Exception:
            # Final fallback if everything else fails
            return 15  # Moderate default

    def _calculate_travel_time(self, distance, traffic_volume):
        """
        Calculate travel time based on distance and traffic volume.
        Uses the quadratic formula to determine speed from traffic volume,
        then converts to travel time.
        """
        a, b, c = -1.4648375, 93.75, -traffic_volume
        d = b * b - (4 * a * c)
        speed = (-b - math.sqrt(d)) / (2 * a)  # km/h
        speed = min(speed, 60)  # Cap speed at 60 km/h
        speed = max(speed, 5)  # Minimum speed of 5 km/h

        # Convert to minutes and add 30 seconds for intersection delay
        travel_time = (distance / speed) * 60 + 30 / 60
        return travel_time

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
        has_timestamp = bool(self.initial_timestamp)

        return (
            f"GraphProblem(initial={self.initial}, "
            f"goals={self.goals}, "
            f"nodes={nodes_count}, "
            f"edges={edges_count}, "
            f"has_locations={has_locations}, "
            f"has_timestamp={has_timestamp})"
        )

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

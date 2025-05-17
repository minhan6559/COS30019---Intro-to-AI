import copy
import math
import heapq
from datetime import datetime, timedelta
from src.graph.node import Node
from src.graph.graph import Graph
from src.search_algorithm.search_algorithm import AStarSearch
import inspect


class YensKShortestPaths:
    """
    Implementation of Yen's algorithm for finding k shortest paths in a graph.
    Uses A* as the base search algorithm.
    """

    def __init__(self):
        self.base_search = AStarSearch()
        self.path_cache = {}  # Cache for paths between nodes
        self.cost_cache = {}  # Cache for path costs

    def find_k_shortest_paths(self, problem, k=5):
        """
        Find k shortest paths from initial to goal using Yen's algorithm.
        Optimized version with caching and efficient data structures.
        """
        # Clear caches for a new search
        self.path_cache = {}
        self.cost_cache = {}

        # Step 1: Find the shortest path with A*
        first_path_node, _, _ = self.base_search.search(problem)

        if not first_path_node:
            return []  # No path found

        # Initialize the list of shortest paths found
        first_path = first_path_node.path_states()
        first_cost = first_path_node.path_cost
        first_info = first_path_node.path_info()
        shortest_paths = [(first_path, first_cost, first_info)]

        # Use a set for fast path checking
        path_set = {tuple(first_path)}

        # Use a heap for potential paths instead of list + sort
        potential_paths = []
        path_id = 0  # For tiebreaking in the heap

        # Temporary storage for edge removals to avoid modifying the graph
        edge_blacklist = set()

        # Main algorithm loop - find k-1 more paths
        for i in range(1, k):
            prev_path = shortest_paths[i - 1][0]

            # For each node in the previous shortest path except the last
            for j in range(
                min(len(prev_path) - 1, 10)
            ):  # Limit deviation points for speed
                spur_node = prev_path[j]
                root_path = prev_path[: j + 1]
                root_path_tuple = tuple(root_path)

                # Clear blacklist for this spur node
                edge_blacklist.clear()

                # Block edges that would lead to previously found paths
                for path, _, _ in shortest_paths:
                    if len(path) > j and path[: j + 1] == root_path:
                        if j + 1 < len(path):
                            edge_blacklist.add((path[j], path[j + 1]))

                # Block edges from the root path to force new paths
                for node_idx in range(j):
                    for neighbor in problem.get_neighbors(prev_path[node_idx]):
                        if neighbor not in root_path[: node_idx + 1]:
                            edge_blacklist.add((prev_path[node_idx], neighbor))

                # Create a subproblem with the spur node as the initial state
                # Using a wrapper instead of deep copy
                spur_problem = self._create_lightweight_subproblem(
                    problem, spur_node, problem.goals, edge_blacklist
                )

                # Try to find a spur path with the filtered graph
                spur_path_node, _, _ = self.base_search.search(spur_problem)

                # If a spur path was found
                if spur_path_node:
                    # Combine the root path and spur path
                    spur_path = spur_path_node.path_states()

                    # Skip the spur node to avoid duplication
                    if len(spur_path) > 1:
                        spur_path = spur_path[1:]

                    total_path = root_path + spur_path
                    total_path_tuple = tuple(total_path)

                    # Check if this path is unique before calculating cost
                    if total_path_tuple not in path_set:
                        path_set.add(total_path_tuple)

                        # Calculate the path cost (using cache when possible)
                        total_cost = self._calculate_cached_cost(problem, total_path)

                        # Calculate path info
                        path_info = self._calculate_cached_path_info(
                            problem, total_path, total_cost
                        )

                        # Add to potential paths using heap
                        heapq.heappush(
                            potential_paths,
                            (total_cost, path_id, (total_path, total_cost, path_info)),
                        )
                        path_id += 1

            # If no more potential paths, break
            if not potential_paths:
                break

            # Get the best path from the heap
            _, _, next_path = heapq.heappop(potential_paths)
            shortest_paths.append(next_path)

        return shortest_paths

    def _create_lightweight_subproblem(
        self, problem, new_initial, goals, edge_blacklist
    ):
        """
        Create a lightweight wrapper problem with a filtered graph.
        Avoids deep copying the entire problem.
        """

        class ProblemWrapper:
            def __init__(self, orig_prob, new_init, new_goals, blacklist):
                self.original = orig_prob
                self.initial = new_init
                self.goals = new_goals
                self.edge_blacklist = blacklist

                # Copy time-dependent attributes if they exist
                if hasattr(orig_prob, "initial_timestamp"):
                    self.initial_timestamp = orig_prob.initial_timestamp

                # Transfer other needed attributes by reference
                for attr in [
                    "locations",
                    "graph",
                    "traffic_volume_lookups",
                    "connections",
                    "prediction_model",
                    "connection_lookup",
                ]:
                    if hasattr(orig_prob, attr):
                        setattr(self, attr, getattr(orig_prob, attr))

                # Set the appropriate initial timestamp based on path to new_init
                if (
                    hasattr(orig_prob, "initial_timestamp")
                    and orig_prob.initial_timestamp
                ):
                    time_elapsed = self._get_time_to_node(orig_prob, new_init)
                    if time_elapsed is not None:
                        if isinstance(orig_prob.initial_timestamp, str):
                            initial_dt = datetime.strptime(
                                orig_prob.initial_timestamp, "%Y-%m-%d %H:%M"
                            )
                        else:
                            initial_dt = orig_prob.initial_timestamp
                        self.initial_timestamp = initial_dt + timedelta(
                            minutes=time_elapsed
                        )

            def _get_time_to_node(self, orig_prob, target_node):
                """Estimate time to reach the target node from initial"""
                if orig_prob.initial == target_node:
                    return 0
                # Simple estimation based on distance and speed
                if hasattr(orig_prob, "locations") and orig_prob.locations:
                    if (
                        orig_prob.initial in orig_prob.locations
                        and target_node in orig_prob.locations
                    ):
                        from haversine import haversine

                        start_coords = orig_prob.locations[orig_prob.initial]
                        target_coords = orig_prob.locations[target_node]
                        dist_km = haversine(start_coords, target_coords, unit="km")
                        # Assuming average speed of 30 km/h
                        return dist_km / 30 * 60
                return None

            def get_neighbors(self, node):
                """Get neighbors, filtering out blacklisted edges"""
                neighbors = self.original.get_neighbors(node)
                return [n for n in neighbors if (node, n) not in self.edge_blacklist]

            def path_cost(self, *args):
                return self.original.path_cost(*args)

            def goal_test(self, state):
                return state in self.goals

            def h(self, node):
                return self.original.h(node)

            def get_step_info(self, *args):
                return self.original.get_step_info(*args)

            # Other necessary method delegations
            def __getattr__(self, name):
                return getattr(self.original, name)

        return ProblemWrapper(problem, new_initial, goals, edge_blacklist)

    def _calculate_cached_cost(self, problem, path):
        """Calculate path cost with caching"""
        path_tuple = tuple(path)
        if path_tuple in self.cost_cache:
            return self.cost_cache[path_tuple]

        total_cost = 0
        current_time = None

        if hasattr(problem, "initial_timestamp"):
            if isinstance(problem.initial_timestamp, str):
                current_time = datetime.strptime(
                    problem.initial_timestamp, "%Y-%m-%d %H:%M"
                )
            else:
                current_time = problem.initial_timestamp

        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]

            # Check for cached segment cost
            segment_key = (from_node, to_node, total_cost)
            if segment_key in self.cost_cache:
                segment_cost = self.cost_cache[segment_key]
            else:
                if (
                    current_time
                    and hasattr(problem, "path_cost")
                    and len(inspect.signature(problem.path_cost).parameters) > 3
                ):
                    segment_cost = problem.path_cost(
                        total_cost, from_node, to_node, current_time
                    )
                else:
                    segment_cost = problem.path_cost(total_cost, from_node, to_node)
                self.cost_cache[segment_key] = segment_cost

            # Update current time for time-dependent costs
            if current_time:
                current_time = current_time + timedelta(
                    minutes=segment_cost - total_cost
                )

            total_cost = segment_cost

        # Cache the total path cost
        self.cost_cache[path_tuple] = total_cost
        return total_cost

    def _calculate_cached_path_info(self, problem, path, precomputed_cost=None):
        """Calculate path info with caching"""
        path_info = []
        current_time = None
        total_cost = 0

        if precomputed_cost is not None:
            # We already calculated this path's cost, no need to recompute
            # Just need to gather the step info
            if hasattr(problem, "initial_timestamp"):
                if isinstance(problem.initial_timestamp, str):
                    current_time = datetime.strptime(
                        problem.initial_timestamp, "%Y-%m-%d %H:%M"
                    )
                else:
                    current_time = problem.initial_timestamp

            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]

                # Get step info if available
                if hasattr(problem, "get_step_info"):
                    step_info = problem.get_step_info(from_node, to_node, total_cost)
                    if step_info:
                        path_info.append(step_info)

                # Update time and cost for next segment
                segment_key = (from_node, to_node, total_cost)
                if segment_key in self.cost_cache:
                    segment_cost = self.cost_cache[segment_key]
                else:
                    if (
                        current_time
                        and hasattr(problem, "path_cost")
                        and len(inspect.signature(problem.path_cost).parameters) > 3
                    ):
                        segment_cost = problem.path_cost(
                            total_cost, from_node, to_node, current_time
                        )
                    else:
                        segment_cost = problem.path_cost(total_cost, from_node, to_node)
                    self.cost_cache[segment_key] = segment_cost

                if current_time:
                    current_time = current_time + timedelta(
                        minutes=segment_cost - total_cost
                    )
                total_cost = segment_cost

        return path_info

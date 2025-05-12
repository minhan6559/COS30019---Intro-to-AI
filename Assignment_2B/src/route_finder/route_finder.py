import random

from src.problem.vicroads_graph_problem import VicRoadsGraphProblem
from src.search_algorithm.search_algorithm import (
    BreadthFirstSearch,
    DepthFirstSearch,
    AStarSearch,
    GreedyBestFirstSearch,
    UniformCostSearch,
    BULBSearch,
)
from src.graph.graph import Graph
from datetime import datetime, time, date, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import math


class RouteFinder:
    """
    Class for finding optimal routes between SCATS sites using various search algorithms
    """

    # Class-level variable to cache the lookups across instances
    _traffic_volume_lookups_cache = None

    def __init__(self, network):
        """
        Initialize with a SiteNetwork object
        """
        self.network = network
        self.graph_problem = None

        # Use cached lookups if available, otherwise create them
        if RouteFinder._traffic_volume_lookups_cache is None:
            RouteFinder._traffic_volume_lookups_cache = self._load_and_preprocess_data()

        # Reference the cached lookups (no copying needed)
        self.traffic_volume_lookups = RouteFinder._traffic_volume_lookups_cache

    def _get_algorithm(self, name):
        """
        Get the specified search algorithm instance
        """
        if name == "A*":
            return AStarSearch()
        elif name == "DFS":
            return DepthFirstSearch()
        elif name == "BFS":
            return BreadthFirstSearch()
        elif name == "GBFS":
            return GreedyBestFirstSearch()
        elif name == "UCS":
            return UniformCostSearch()
        elif name == "BULB":
            return BULBSearch()
        return None

    def _load_and_preprocess_data(self):
        """
        Load dataframes for October and November and convert them to nested dictionaries
        for faster lookup by model_name, location, date_str, interval_id
        """
        model_dataframes = {
            "LSTM": pd.read_csv(
                "processed_data/complete_csv_oct_nov_2006/lstm_model/lstm_model_complete_data.csv"
            ),
            "GRU": pd.read_csv(
                "processed_data/complete_csv_oct_nov_2006/gru_model/gru_model_complete_data.csv"
            ),
            "CNN_LSTM": pd.read_csv(
                "processed_data/complete_csv_oct_nov_2006/cnn_lstm_model/cnn_lstm_model_complete_data.csv"
            ),
        }

        # Convert dataframes to nested dictionaries for faster lookup
        lookup_dictionaries = {}

        for model_name, df in model_dataframes.items():
            # Process Location column to ensure consistent formatting
            df["Location"] = df["Location"].str.strip()

            # Convert traffic_volume to int type
            df["traffic_volume"] = df["traffic_volume"].astype(int)

            # Create multi-level dictionary using Pandas' to_dict method for faster conversion
            # New structure: {location: {date_str: {interval_id: traffic_volume}}}
            location_dict = {}

            # Group by Location first
            location_groups = df.groupby("Location")
            for location, location_group in location_groups:
                date_dict = {}

                # For each location, group by Date
                date_groups = location_group.groupby("Date")
                for date_str, date_group in date_groups:
                    # For each date, create interval_id to traffic_volume mapping
                    interval_traffic_dict = date_group.set_index("interval_id")[
                        "traffic_volume"
                    ].to_dict()
                    date_dict[date_str] = interval_traffic_dict

                location_dict[location] = date_dict

            lookup_dictionaries[model_name] = location_dict

        return lookup_dictionaries

    def _create_search_graph(
        self, origin, destination, prediction_model="LSTM", datetime_str=None
    ):
        """
        Convert SiteNetwork to SearchGraph format for search algorithms
        datetime_str is in the format like this "2006-11-30 14:15"
        """
        graph_dict = {}
        locations = {}

        # Add all site IDs as nodes with their coordinates
        for site_id, site in self.network.sites_data.items():
            site_id = int(site_id)

            locations[site_id] = (site["latitude"], site["longitude"])

            # Initialize empty dict for each node
            if site_id not in graph_dict:
                graph_dict[site_id] = {}

        # Add connections as edges with placeholder costs (-1)
        # Actual costs will be calculated dynamically based on time
        for conn in self.network.connections:
            from_id = int(conn["from_id"])
            to_id = int(conn["to_id"])

            # Add edge to the graph with placeholder cost
            if from_id not in graph_dict:
                graph_dict[from_id] = {}

            # Use -1 as a placeholder, actual cost will be calculated in path_cost
            graph_dict[from_id][to_id] = -1

        # Create the graph problem with time-dependent information
        graph_problem = VicRoadsGraphProblem(
            origin,
            destination,
            Graph(graph_dict),
            locations,
            initial_timestamp=datetime_str,
            traffic_volume_lookups=self.traffic_volume_lookups,
            connections=self.network.connections,
            prediction_model=prediction_model,
        )

        # Return a dummy traffic_volume_lookup (not used anymore) and the graph problem
        return {}, graph_problem

    def _calculate_travel_time(self, distance, traffic_volume):
        """
        Calculate travel time based on distance and traffic volume
        """
        a, b, c = -1.4648375, 93.75, -traffic_volume
        d = b * b - (4 * a * c)
        speed = (-b + math.sqrt(d)) / (2 * a)  # km/h
        speed = min(speed, 60)  # Cap speed at 60 km/h
        speed = max(speed, 5)  # Minimum speed of 5 km/h

        # Convert to minutes and add 30 seconds for intersection delay
        travel_time = (distance / speed) * 60 + 30 / 60
        return travel_time

    def find_multiple_routes(
        self,
        origin_id,
        destination_id,
        selected_algorithms=None,
        prediction_model="LSTM",
        datetime_str=None,
    ):
        """
        Find routes from origin to destination using multiple algorithms
        Returns a list of routes with their details

        datetime_str is in the format like this "2006-11-30 14:15"
        """
        # Use all algorithms if none specified
        all_algorithms = ["A*", "DFS", "BFS", "GBFS", "UCS", "BULB"]
        if selected_algorithms is None or "All" in selected_algorithms:
            selected_algorithms = all_algorithms

        # Create the graph ONCE for all algorithms
        # Note: traffic_volume_lookup is now a dummy and not used
        _, self.graph_problem = self._create_search_graph(
            origin_id, destination_id, prediction_model, datetime_str
        )

        routes = []

        # Run each selected algorithm
        for alg_name in selected_algorithms:
            # Use an empty dictionary as traffic_volume_lookup (not used anymore)
            dummy_traffic_lookup = {}
            path, total_cost, route_info = self.find_best_route(
                alg_name, dummy_traffic_lookup
            )

            if path:
                routes.append(
                    {
                        "algorithm": alg_name,
                        "path": path,
                        "total_cost": total_cost,
                        "route_info": route_info,
                        "traffic_level": "",  # Will assign later
                        "prediction_model": prediction_model,
                        "datetime": datetime_str,
                    }
                )

        # Sort routes by total cost (travel time)
        routes.sort(key=lambda x: x["total_cost"])

        # Assign colors based on relative performance (best to worst)
        route_colors = ["aqua", "blue", "green", "orange", "red", "darkred"]
        route_descriptions = [
            "Best route",
            "Second best",
            "Third best",
            "Fourth best",
            "Fifth best",
            "Worst route",
        ]

        for i, route in enumerate(routes[:6]):
            color_index = min(i, len(route_colors) - 1)
            route["traffic_level"] = route_colors[color_index]
            route["route_rank"] = route_descriptions[color_index]

        # Limit to at most 6 routes
        return routes[:6]

    def find_best_route(self, algorithm, traffic_volume_lookup):
        """
        Find the best route using an already created graph
        """
        # Get the search algorithm instance
        search_alg = self._get_algorithm(algorithm)

        if not search_alg:
            return None, None, None

        # Execute search algorithm
        dest_node, nodes_expanded, nodes_created = search_alg.search(self.graph_problem)

        # If no path is found, return None
        if not dest_node:
            return None, None, None

        path = dest_node.path_states()

        # Calculate route information
        return path, *self._calculate_route_details(path, traffic_volume_lookup)

    def _calculate_route_details(self, path, traffic_volume_lookup):
        """
        Calculate total travel time and create a list of steps for a path
        """
        total_cost = 0
        route_info = []
        cumulative_time = 0

        # Get the datetime object for the initial timestamp
        datetime_str = self.graph_problem.initial_timestamp
        if datetime_str:
            try:
                initial_dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                initial_dt = datetime_str  # In case it's already a datetime object
        else:
            # Default to current time if no timestamp provided
            initial_dt = datetime.now()

        for i in range(len(path) - 1):
            from_id = path[i]
            to_id = path[i + 1]

            # Find the connection between these sites
            connection = self._find_connection(from_id, to_id)

            if connection:
                # Calculate the current time for this leg
                current_time = initial_dt + timedelta(minutes=cumulative_time)

                # Format for lookup
                date_str = current_time.date().strftime("%Y-%m-%d")
                hour = current_time.hour
                minute = current_time.minute
                interval_id = (hour * 4) + (minute // 15)

                # Get traffic volume for this leg at the current time using the fallback mechanism
                location = connection.get("approach_location", "").strip()

                if (
                    self.graph_problem.traffic_volume_lookups
                    and self.graph_problem.prediction_model
                ):
                    traffic_volume = (
                        self.graph_problem._get_traffic_volume_with_fallback(
                            location, date_str, interval_id
                        )
                    )
                else:
                    # Use reasonable default if no traffic data available
                    traffic_volume = 15

                # Calculate travel time for this leg
                distance = connection["distance"]
                travel_time = self.graph_problem._calculate_travel_time(
                    distance, traffic_volume
                )

                # Update cumulative time
                cumulative_time += travel_time

                # Add to total cost
                total_cost += travel_time

                # Add step info
                route_info.append(
                    {
                        "from_id": from_id,
                        "to_id": to_id,
                        "road": connection["shared_road"],
                        "distance": connection["distance"],
                        "travel_time": travel_time,
                        "from_lat": connection["from_lat"],
                        "from_lng": connection["from_lng"],
                        "to_lat": connection["to_lat"],
                        "to_lng": connection["to_lng"],
                        "traffic_volume": traffic_volume,
                        "time_of_day": current_time.strftime("%H:%M"),
                        "date": current_time.strftime("%Y-%m-%d"),
                    }
                )

        return total_cost, route_info

    def _find_connection(self, from_id, to_id):
        """
        Find a connection between two sites
        """
        for conn in self.network.connections:
            if conn["from_id"] == from_id and conn["to_id"] == to_id:
                return conn
        return None

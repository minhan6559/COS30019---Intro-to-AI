import random
import os
import pickle

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
from src.search_algorithm.yens_algorithm import YensKShortestPaths
import inspect


class RouteFinder:
    """
    Class for finding optimal routes between SCATS sites using various search algorithms
    """

    # Class-level variable to cache the lookups across instances
    _traffic_volume_lookups_cache = None

    # Path for the pickled lookup data
    PICKLE_PATH = "processed_data/complete_csv_oct_nov_2006/traffic_volume_lookups.pkl"

    def __init__(self, network):
        """
        Initialize with a SiteNetwork object
        """
        self.network = network
        self.graph_problem = None

        # Use cached lookups if available, otherwise create them
        if RouteFinder._traffic_volume_lookups_cache is None:
            # Try to load from pickle file first
            if self._load_from_pickle():
                print("Loaded traffic volume lookups from pickle file")
            else:
                print("Creating traffic volume lookups for the first time")
                RouteFinder._traffic_volume_lookups_cache = (
                    self._load_and_preprocess_data()
                )
                # Save to pickle for future use
                self._save_to_pickle()

        # Reference the cached lookups (no copying needed)
        self.traffic_volume_lookups = RouteFinder._traffic_volume_lookups_cache

    def _load_from_pickle(self):
        """Try to load the traffic volume lookups from pickle file"""
        try:
            if os.path.exists(self.PICKLE_PATH):
                with open(self.PICKLE_PATH, "rb") as f:
                    RouteFinder._traffic_volume_lookups_cache = pickle.load(f)
                return True
            return False
        except Exception as e:
            print(f"Error loading pickle file: {str(e)}")
            return False

    def _save_to_pickle(self):
        """Save the traffic volume lookups to pickle file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.PICKLE_PATH), exist_ok=True)
            with open(self.PICKLE_PATH, "wb") as f:
                pickle.dump(RouteFinder._traffic_volume_lookups_cache, f)
            print(f"Saved traffic volume lookups to {self.PICKLE_PATH}")
        except Exception as e:
            print(f"Error saving pickle file: {str(e)}")

    def _get_algorithm(self, name):
        """
        Get the specified search algorithm instance
        """
        algorithms_lookup = {
            "A*": AStarSearch(),
            "DFS": DepthFirstSearch(),
            "BFS": BreadthFirstSearch(),
            "GBFS": GreedyBestFirstSearch(),
            "UCS": UniformCostSearch(),
            "BULB": BULBSearch(),
        }
        return algorithms_lookup.get(name)

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

        return graph_problem

    def _calculate_travel_time(self, distance, traffic_volume):
        """
        Calculate travel time based on distance and traffic volume
        """
        a, b, c = -1.4648375, 93.75, -traffic_volume
        d = b * b - (4 * a * c)
        speed = (-b - math.sqrt(d)) / (2 * a)  # km/h
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
        all_algorithms = ["A*", "UCS", "BULB", "GBFS", "DFS", "BFS"]
        if selected_algorithms is None or "All" in selected_algorithms:
            selected_algorithms = all_algorithms

        # Create the graph ONCE for all algorithms
        self.graph_problem = self._create_search_graph(
            origin_id, destination_id, prediction_model, datetime_str
        )

        routes = []

        # Run each selected algorithm
        for alg_name in selected_algorithms:
            # Use an empty dictionary as traffic_volume_lookup (not used anymore)
            path, total_cost, route_info = self.find_best_route(alg_name)

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
        route_colors = [
            "aqua",
            "blue",
            "green",
            "orange",
            "red",
            "darkred",
            "purple",
            "pink",
        ]
        route_descriptions = [
            "Best route",
            "Second best",
            "Third best",
            "Fourth best",
            "Fifth best",
            "Sixth best",
            "Seventh best",
            "Eighth best",
        ]

        for i, route in enumerate(routes[:8]):
            color_index = min(i, len(route_colors) - 1)
            route["traffic_level"] = route_colors[color_index]
            route["route_rank"] = route_descriptions[color_index]

        # Limit to at most 8 routes
        return routes[:8]

    def find_best_route(self, algorithm):
        """
        Find the best route using an already created graph
        """
        # Get the search algorithm instance
        search_alg = self._get_algorithm(algorithm)

        if not search_alg:
            return None, None, None

        # Execute search algorithm
        dest_node, _, _ = search_alg.search(self.graph_problem)

        # If no path is found, return None
        if not dest_node:
            return None, None, None

        path = dest_node.path_states()
        total_cost = dest_node.path_cost
        route_info = dest_node.path_info()

        return path, total_cost, route_info

    def find_top_k_routes(
        self,
        origin_id,
        destination_id,
        k=5,
        prediction_model="LSTM",
        datetime_str=None,
    ):
        """
        Find the top K shortest paths from origin to destination using Yen's algorithm.

        Args:
            origin_id: ID of the origin node
            destination_id: ID of the destination node
            k: Number of paths to find (default 5)
            prediction_model: Traffic prediction model to use (default "LSTM")
            datetime_str: Date and time string in format "YYYY-MM-DD HH:MM"

        Returns:
            List of routes with their details, sorted by cost (fastest first)
        """
        # Create the graph for search
        self.graph_problem = self._create_search_graph(
            origin_id, destination_id, prediction_model, datetime_str
        )

        # Create Yen's algorithm instance
        yens_algorithm = YensKShortestPaths()

        # Find k shortest paths
        shortest_paths = yens_algorithm.find_k_shortest_paths(self.graph_problem, k)

        # Convert to the same format as find_multiple_routes
        routes = []
        for i, (path, total_cost, route_info) in enumerate(shortest_paths):
            routes.append(
                {
                    "algorithm": f"Yen-{i+1}",
                    "path": path,
                    "total_cost": total_cost,
                    "route_info": route_info,
                    "traffic_level": "",
                    "prediction_model": prediction_model,
                    "datetime": datetime_str,
                }
            )

        # Sort routes by total cost (travel time)
        routes.sort(key=lambda x: x["total_cost"])

        # Assign colors based on relative performance (best to worst)
        route_colors = [
            "aqua",
            "blue",
            "green",
            "orange",
            "red",
            "darkred",
            "purple",
            "pink",
        ]
        route_descriptions = [
            "Best route",
            "Second best",
            "Third best",
            "Fourth best",
            "Fifth best",
            "Sixth best",
            "Seventh best",
            "Eighth best",
        ]

        for i, route in enumerate(routes[:8]):
            color_index = min(i, len(route_colors) - 1)
            route["traffic_level"] = route_colors[color_index]
            route["route_rank"] = route_descriptions[color_index]

        # Limit to at most 8 routes
        return routes[: min(k, 8)]

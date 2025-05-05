import pandas as pd
import folium
from folium.plugins import AntPath
from src.visualizer.base_visualizer import BaseVisualizer


class RouteVisualizer(BaseVisualizer):
    """
    Specialized class for visualizing routes between SCATS sites
    """

    def create_route_map(self, route_info, traffic_level=None):
        """
        Create a folium map with the route highlighted
        traffic_level: color string like "green", "yellow", "orange", etc.
        """
        if not route_info:
            return None

        # Calculate map center based on the route
        lats = [step["from_lat"] for step in route_info] + [route_info[-1]["to_lat"]]
        lngs = [step["from_lng"] for step in route_info] + [route_info[-1]["to_lng"]]
        map_center = [sum(lats) / len(lats), sum(lngs) / len(lngs)]

        # Create the map with centered view on the route
        m = self._create_base_map(center=map_center, zoom_start=13)

        # Add all sites to the map as background
        self._add_background_sites(m)

        # Add route points with different color
        self._add_route_points(m, route_info)

        # Add route path
        self._add_route_path(m, route_info, traffic_level)

        return m

    def _add_route_points(self, m, route_info):
        """
        Add route points to the map with appropriate colors
        """
        route_sites = set(
            [step["from_id"] for step in route_info] + [route_info[-1]["to_id"]]
        )
        for site_id in route_sites:
            site = self.network.get_site(site_id)
            if site:
                # Determine point color based on role in the route
                if site_id == int(route_info[0]["from_id"]):
                    color = "green"  # Origin
                    radius = 10
                elif site_id == int(route_info[-1]["to_id"]):
                    color = "red"  # Destination
                    radius = 10
                else:
                    color = "blue"  # Intermediate points
                    radius = 8

                folium.CircleMarker(
                    location=[site["latitude"], site["longitude"]],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"Site ID: {site_id}<br>Roads: {', '.join(site['connected_roads'])}",
                ).add_to(m)

    def _add_route_path(self, m, route_info, traffic_level=None):
        """
        Add route path to the map with the specified color
        """
        route_coords = []
        for step in route_info:
            route_coords.append([step["from_lat"], step["from_lng"]])
        route_coords.append([route_info[-1]["to_lat"], route_info[-1]["to_lng"]])

        # Use traffic_level directly as the color if provided, otherwise use default
        path_color = traffic_level if traffic_level else "blue"

        AntPath(
            locations=route_coords,
            color=path_color,
            weight=4,
            opacity=0.8,
            tooltip="Route",
        ).add_to(m)

    def create_multi_route_map(self, routes):
        """
        Create a map showing multiple routes with different colors
        """
        if not routes:
            return None

        # Get all route coordinates to calculate map center
        all_lats = []
        all_lngs = []

        for route in routes:
            route_info = route["route_info"]
            lats = [step["from_lat"] for step in route_info] + [
                route_info[-1]["to_lat"]
            ]
            lngs = [step["from_lng"] for step in route_info] + [
                route_info[-1]["to_lng"]
            ]
            all_lats.extend(lats)
            all_lngs.extend(lngs)

        map_center = [sum(all_lats) / len(all_lats), sum(all_lngs) / len(all_lngs)]

        # Create the map centered on all routes
        m = self._create_base_map(center=map_center, zoom_start=13)

        # Add all sites to the map as background
        self._add_background_sites(m)

        # Add route points for all routes
        self._add_multi_route_points(m, routes)

        # Add each route with its own color - draw in reverse order so best routes are on top
        self._add_multi_route_paths(m, routes)

        return m

    def _add_multi_route_points(self, m, routes):
        """
        Add route points for multiple routes with appropriate colors
        """
        # Track all sites on any route
        all_route_sites = set()
        for route in routes:
            route_info = route["route_info"]
            route_sites = set(
                [step["from_id"] for step in route_info] + [route_info[-1]["to_id"]]
            )
            all_route_sites.update(route_sites)

        # Add route points
        for site_id in all_route_sites:
            site = self.network.get_site(site_id)
            if site:
                # Check if site is origin or destination
                is_origin = any(
                    route["route_info"][0]["from_id"] == site_id for route in routes
                )
                is_destination = any(
                    route["route_info"][-1]["to_id"] == site_id for route in routes
                )

                if is_origin:
                    color = "green"  # Origin
                    radius = 10
                elif is_destination:
                    color = "red"  # Destination
                    radius = 10
                else:
                    color = "blue"  # Intermediate points
                    radius = 8

                folium.CircleMarker(
                    location=[site["latitude"], site["longitude"]],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"Site ID: {site_id}<br>Roads: {', '.join(site['connected_roads'])}",
                ).add_to(m)

    def _add_multi_route_paths(self, m, routes):
        """
        Add paths for multiple routes with appropriate colors
        """
        # Add each route with its own color - draw in reverse order so best routes are on top
        for route in reversed(routes):
            route_info = route["route_info"]
            traffic_level = route[
                "traffic_level"
            ]  # This should be "green", "yellow", "orange", etc.
            algorithm = route["algorithm"]

            # Create route coordinates
            route_coords = []
            for step in route_info:
                route_coords.append([step["from_lat"], step["from_lng"]])
            route_coords.append([route_info[-1]["to_lat"], route_info[-1]["to_lng"]])

            # Use the traffic level color directly
            path_color = traffic_level

            # Create the path with a tooltip showing algorithm and travel time
            tooltip = f"{algorithm} - Travel time: {route['total_cost']:.2f} minutes"

            AntPath(
                locations=route_coords,
                color=path_color,
                weight=4,
                opacity=0.8,
                tooltip=tooltip,
            ).add_to(m)

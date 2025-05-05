import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
import time as time_module  # Changed this to avoid conflict
from src.views.base_page import BasePage
from datetime import datetime, time, date  # Add this line


class RoutePage(BasePage):
    """
    Page for finding optimal routes between SCATS sites
    """

    def __init__(self, network, route_finder, visualizer):
        """
        Initialize with a SiteNetwork, RouteFinder, and NetworkVisualizer
        """
        super().__init__(network)
        self.route_finder = route_finder
        self.visualizer = visualizer

    def render(self):
        """
        Render the route finding page
        """
        st.title("Traffic-Based Route Guidance System (TBRGS)")
        st.header("Route Finder")

        # Get sorted list of site IDs for dropdowns
        site_ids = sorted(list(map(int, self.network.sites_data.keys())))

        # Create two columns for origin and destination selection
        origin_id, destination_id = self._render_site_selection(site_ids)

        # Create two columns for algorithm and model selection
        col1, col2 = st.columns(2)

        with col1:
            # Algorithm selection
            algorithm_options = ["A*", "DFS", "BFS", "GBFS", "UCS", "All"]
            selected_algorithms = st.multiselect(
                "Select algorithms to use (or select 'All'):",
                options=algorithm_options,
                default=["All"],
            )

        with col2:
            # Model selection
            model_options = ["LSTM", "GRU", "CNN_LSTM"]
            selected_model = st.selectbox(
                "Select prediction model:", options=model_options, index=0
            )

        # Create another row with two columns for date and time selection
        col3, col4 = st.columns(2)

        with col3:
            # Date selection with constraint
            selected_date = st.date_input(
                "Select date and time:",
                value=datetime(2006, 10, 1).date(),
                min_value=datetime(2006, 10, 1).date(),
                max_value=datetime(2006, 11, 30).date(),
            )

        with col4:
            # st.write("Select time:")
            # Create two columns for hour and minute selection
            hour_col, minute_col = st.columns(2)

            with hour_col:
                # Hour selection (0-23)
                selected_hour = st.selectbox(
                    "Hour",
                    options=list(range(24)),
                    format_func=lambda x: f"{x:02d}",
                    index=8,  # Default to 8:00
                )

            with minute_col:
                # Minute selection (0-59)
                selected_minute = st.selectbox(
                    "Minute",
                    options=list(range(60)),
                    format_func=lambda x: f"{x:02d}",
                    index=0,  # Default to 8:00
                )

            # Combine hour and minute into time object
            selected_time = time(hour=selected_hour, minute=selected_minute)

            # Show the actual rounded time that will be used
            rounded_time = self._round_to_15_minutes(selected_time)
            # st.caption(f"Will be rounded to: {rounded_time.strftime('%H:%M')}")

        # Find route button
        if st.button("Find Routes", type="primary"):
            if not selected_algorithms:
                st.warning("Please select at least one algorithm")
            elif origin_id == destination_id:
                st.error("Origin and destination cannot be the same")
            else:
                # Round time to nearest 15-minute interval
                rounded_time = self._round_to_15_minutes(selected_time)
                datetime_str = f"{selected_date.strftime('%Y-%m-%d')} {rounded_time.strftime('%H:%M')}"

                self._find_and_display_routes(
                    origin_id,
                    destination_id,
                    selected_algorithms,
                    selected_model,
                    datetime_str,
                )

    def _render_site_selection(self, site_ids):
        """
        Render the origin and destination selection UI
        """
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Origin")
            origin_id = st.selectbox(
                "Select origin site:",
                options=site_ids,
                format_func=lambda x: f"Site {x}",
                key="origin",
            )

            # Show origin site details
            if origin_id:
                self.display_site_info(origin_id)

        with col2:
            st.subheader("Destination")
            destination_id = st.selectbox(
                "Select destination site:",
                options=site_ids,
                format_func=lambda x: f"Site {x}",
                key="destination",
            )

            # Show destination site details
            if destination_id:
                self.display_site_info(destination_id)

        return origin_id, destination_id

    def _display_route_results(self, origin_id, destination_id, routes, selected_model):
        """
        Display the results of route finding
        """

        # Display route summary
        st.subheader("Route Summary")

        # Create a summary table
        summary_data = []
        for i, route in enumerate(routes):
            with st.expander(
                f"Route {i+1} ({route['algorithm']} - {route['route_rank']})"
            ):
                summary_data.append(
                    {
                        "Route": i + 1,
                        "Algorithm": route["algorithm"],
                        "Travel Time (min)": f"{route['total_cost']:.2f}",
                        "Intermediate Sites": len(route["path"]) - 2,
                        "Model": selected_model,
                        "Traffic Level": route["traffic_level"].capitalize(),
                    }
                )

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True)

        # Show multi-route map
        st.subheader("Route Map")
        m = self.visualizer.create_multi_route_map(routes)
        folium_static(m)

        # Show detailed information for each route
        st.subheader("Route Details")

        for i, route in enumerate(routes):
            with st.expander(
                f"Route {i+1} ({route['algorithm']} - {route['traffic_level'].capitalize()} Traffic)"
            ):
                # Show path
                path_str = " → ".join([str(node) for node in route["path"]])
                st.code(path_str)

                # Show route details in a table
                route_df = pd.DataFrame(
                    [
                        {
                            "Step": j + 1,
                            "From Site": step["from_id"],
                            "To Site": step["to_id"],
                            "Road": step["road"],
                            "Distance (km)": f"{step['distance']:.2f}",
                            "Travel Time (min)": f"{step['travel_time']:.2f}",
                        }
                        for j, step in enumerate(route["route_info"])
                    ]
                )
                st.dataframe(route_df, hide_index=True)

                # Show individual route map
                st.subheader("Individual Route Map")
                individual_map = self.visualizer.create_route_map(
                    route["route_info"], route["traffic_level"]
                )
                folium_static(individual_map)

    def _round_to_15_minutes(self, time_obj):
        """Round time to the nearest 15-minute interval (going backwards)"""
        # Convert to minutes since midnight
        total_minutes = time_obj.hour * 60 + time_obj.minute

        # Round down to nearest 15-minute interval
        rounded_minutes = (total_minutes // 15) * 15

        # Convert back to time
        hour = rounded_minutes // 60
        minute = rounded_minutes % 60

        return datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()

    def _find_and_display_routes(
        self,
        origin_id,
        destination_id,
        selected_algorithms,
        selected_model,
        datetime_str,
    ):
        """
        Find and display routes between origin and destination using selected algorithms
        """
        with st.spinner(
            f"Finding optimal routes using {selected_model} model for {datetime_str}..."
        ):
            # Add a small delay to show the spinner (for better UX)
            time_module.sleep(0.5)  # Changed to time_module

            # Find routes using selected algorithms
            routes = self.route_finder.find_multiple_routes(
                origin_id,
                destination_id,
                selected_algorithms,
                selected_model,
                datetime_str,
            )

            if routes:
                self._display_route_results(
                    origin_id, destination_id, routes, selected_model
                )
            else:
                st.error(
                    f"No routes found from Site {origin_id} to Site {destination_id}"
                )

    def _render_implementation_notes(self):
        """
        Render implementation notes expander
        """
        with st.expander("Implementation Notes"):
            st.info(
                """
            **Current Implementation Notes:**
            
            - The route finder uses multiple search algorithms:
              * A* (A-star): Optimal path finding using distance heuristic
              * DFS (Depth-First Search): Explores as far as possible along each branch
              * BFS (Breadth-First Search): Explores all neighbors at current depth before moving deeper
              * GBFS (Greedy Best-First Search): Always moves toward the goal using heuristic
              * UCS (Uniform Cost Search): Explores paths in order of increasing cost
            
            - Travel time is currently estimated based on distance with a random traffic factor (1.1-1.5) to simulate varying traffic conditions
            - Route colors indicate traffic conditions: Green (low traffic), Yellow (medium traffic), Red (high traffic)
            - A 30-second delay is added for each intersection as per the assignment requirements
            - In the future, the travel time estimation will be replaced with predictions from machine learning models (LSTM and GRU)
            """
            )

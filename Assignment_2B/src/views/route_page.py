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

        # Create a container with a blue background for the route finder
        with st.container():
            st.markdown(
                """
            <div style='background-color: #f0f5ff; padding: 20px; border-radius: 10px; border: 1px solid #d0e0ff;'>
                <h2 style='text-align: center; color: #0066CC; margin-top: 0;'>Plan Your Route</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Add some space
            st.markdown("<br>", unsafe_allow_html=True)

            # Get sorted list of site IDs for dropdowns
            site_ids = sorted(list(map(int, self.network.sites_data.keys())))

            # Create row with origin, destination, and search method
            origin_id, destination_id, search_method, selected_algorithms, k_value = (
                self._render_site_selection(site_ids)
            )

            # Add space before the button
            st.markdown("<br>", unsafe_allow_html=True)

            # Create three columns for the inputs
            col1, col2, col3 = st.columns(3)

            with col1:
                # Model section title and selection
                st.markdown("#### 🤖 Model")

                # Model selection with better UI
                model_options = ["LSTM", "GRU", "CNN_LSTM"]
                selected_model = st.selectbox(
                    "Select model:",
                    options=model_options,
                    index=0,
                    help="Choose the traffic prediction model to use",
                )

                # Add a brief model description
                model_descriptions = {
                    "LSTM": "Long Short-Term Memory - Good for time series data",
                    "GRU": "Gated Recurrent Unit - Faster but slightly less accurate",
                    "CNN_LSTM": "Combined CNN+LSTM - Best for complex patterns",
                }
                st.caption(model_descriptions[selected_model])

            with col2:
                # Date section title and selection
                st.markdown("#### 📅 Travel Date")

                # Date selection with constraint
                selected_date = st.date_input(
                    "Select date:",
                    value=datetime(2006, 10, 1).date(),
                    min_value=datetime(2006, 10, 1).date(),
                    max_value=datetime(2006, 11, 30).date(),
                )

                # Add note about data availability
                st.caption("Data available only for October-November 2006")

            with col3:
                # Time section title and selection
                st.markdown("#### ⏰ Start Time")

                # Create a container for hour and minute in one row
                time_col1, time_col2 = st.columns(2)

                with time_col1:
                    # Hour selection (0-23)
                    selected_hour = st.selectbox(
                        "Hour:",
                        options=list(range(24)),
                        format_func=lambda x: f"{x:02d}",
                        index=8,  # Default to 08:08
                    )

                with time_col2:
                    # Minute selection (0-59) - Now allowing all minutes, not just 15-minute intervals
                    selected_minute = st.selectbox(
                        "Minute:",
                        options=list(range(60)),  # Full range of minutes
                        format_func=lambda x: f"{x:02d}",
                        index=8,  # Default to 08:08
                    )

                # Combine hour and minute into time object
                selected_time = time(hour=selected_hour, minute=selected_minute)

            # Add space before the button
            st.markdown("<br>", unsafe_allow_html=True)

            # Center the button and make it larger
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                find_routes = st.button(
                    "🔍 Find Optimal Routes",
                    type="primary",
                    use_container_width=True,
                    help="Click to find the best routes between selected sites",
                )

            # Find route button
            if find_routes:
                if search_method == "Traditional Search" and not selected_algorithms:
                    st.warning("⚠️ Please select at least one algorithm")
                elif origin_id == destination_id:
                    st.error("❌ Origin and destination cannot be the same")
                else:
                    datetime_str = f"{selected_date.strftime('%Y-%m-%d')} {selected_time.strftime('%H:%M')}"

                    self._find_and_display_routes(
                        origin_id,
                        destination_id,
                        selected_algorithms,
                        selected_model,
                        datetime_str,
                        search_method,
                        k_value,
                    )

            # Show implementation notes at the bottom of the page
            self._render_implementation_notes()

    def _render_site_selection(self, site_ids):
        """
        Render the origin and destination selection UI and search method
        """
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🚩 Origin")
            origin_id = st.selectbox(
                "Select origin site:",
                options=site_ids,
                format_func=lambda x: f"Site {x}",
                key="origin",
            )

            # Show origin site details
            if origin_id:
                site_data = self.network.get_site(origin_id)
                if site_data:
                    st.markdown(
                        f"""
                        <div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #d0e0ff;'>
                            <strong>Connected roads:</strong> {', '.join(site_data['connected_roads'])}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with col2:
            st.markdown("### 🏁 Destination")
            destination_id = st.selectbox(
                "Select destination site:",
                options=site_ids,
                format_func=lambda x: f"Site {x}",
                key="destination",
            )

            # Show destination site details
            if destination_id:
                site_data = self.network.get_site(destination_id)
                if site_data:
                    st.markdown(
                        f"""
                        <div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #d0e0ff;'>
                            <strong>Connected roads:</strong> {', '.join(site_data['connected_roads'])}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with col3:
            st.markdown("### 🔍 Search Method")
            # Method selection: Yen's K-Shortest or traditional algorithms
            search_method = st.radio(
                "Select search method:",
                options=["K-Shortest", "Traditional Search"],
                index=0,  # Default to Yen's
                help="Choose whether to use Yen's k-shortest paths algorithm or traditional search algorithms",
                horizontal=True,  # Display radio buttons horizontally
            )

            if search_method == "K-Shortest":
                # K selection slider
                k_value = st.slider(
                    "Number of paths (k):",
                    min_value=1,
                    max_value=8,
                    value=5,
                    step=1,
                    help="Select how many alternative paths to find",
                )
                selected_algorithms = None  # Not used with Yen's
            else:
                # Traditional algorithm selection with better UI
                algorithm_options = ["All", "A*", "DFS", "BFS", "GBFS", "UCS"]
                selected_algorithms = st.multiselect(
                    "Select algorithms:",
                    options=algorithm_options,
                    default=["All"],
                    help="Select one or more algorithms or 'All' to compare them",
                )
                k_value = None  # Not used with traditional algorithms

        return origin_id, destination_id, search_method, selected_algorithms, k_value

    def _display_route_results(self, origin_id, destination_id, routes, selected_model):
        """
        Display the results of route finding
        """
        # Add a section divider
        st.markdown("---")

        # Header for results section
        st.markdown(
            f"""
        <h2 style='text-align: center;'>Results: Site {origin_id} → Site {destination_id}</h2>
        """,
            unsafe_allow_html=True,
        )

        # Set custom CSS for map containers to use full width
        st.markdown(
            """
            <style>
            .stfolium-container {
                width: 100% !important;
            }
            iframe {
                width: 100% !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Show multi-route map first - it's the most important visual
        st.subheader("🗺️ Route Comparison Map")
        m = self.visualizer.create_multi_route_map(routes)
        folium_static(m, width=1200)

        # Create route summary cards in a grid
        st.subheader("🔍 Route Summary")

        # Create a grid of cards using columns (4 columns)
        cols = st.columns(4)

        for i, route in enumerate(routes):
            with cols[i % 4]:
                # Create a card-like container
                traffic_color = route["traffic_level"].lower()
                card_bg_color = {
                    "aqua": "#e6f7ff",
                    "blue": "#e6f0ff",
                    "green": "#e6ffe6",
                    "orange": "#fff7e6",
                    "red": "#ffe6e6",
                    "darkred": "#ffcccc",
                    "purple": "#f3e6ff",
                    "pink": "#ffe6f7",
                    "black": "#f0f0f0",
                    "gray": "#f5f5f5",
                }.get(traffic_color, "#f0f0f0")

                st.markdown(
                    f"""
                <div style='background-color: {card_bg_color}; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #ddd;'>
                    <h4 style='margin-top: 0;'>Route {i+1}: {route['algorithm']}</h4>
                    <p><strong>Travel Time:</strong> {route['total_cost']:.2f} min</p>
                    <p><strong>Path Color:</strong> {route['traffic_level'].capitalize()}</p>
                    <p><strong>Intermediate Sites:</strong> {len(route['path']) - 2}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Show detailed information for each route
        st.subheader("📋 Route Details")

        # Use tabs for detailed route information
        route_tabs = st.tabs(
            [f"Route {i+1}: {r['algorithm']}" for i, r in enumerate(routes)]
        )

        # Add custom CSS for code wrapping
        st.markdown(
            """
            <style>
            .path-code {
                white-space: normal !important;
                overflow-wrap: break-word !important;
                word-wrap: break-word !important;
                overflow-x: hidden;
            }
            pre {
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
            }
            code {
                white-space: pre-wrap !important;
                word-break: break-all !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        for i, (tab, route) in enumerate(zip(route_tabs, routes)):
            with tab:
                # Create two columns
                col1, col2 = st.columns([2, 3])

                with col1:
                    # Show stats
                    st.markdown(f"**Algorithm:** {route['algorithm']}")
                    st.markdown(f"**Travel Time:** {route['total_cost']:.2f} minutes")
                    st.markdown(
                        f"**Path Color:** {route['traffic_level'].capitalize()}"
                    )
                    st.markdown(f"**Model Used:** {selected_model}")

                    # Show path with wrapping
                    st.markdown("**Complete Path:**")
                    path_str = " → ".join([str(node) for node in route["path"]])

                    # Use HTML to make it wrap properly
                    st.markdown(
                        f"""<div class="path-code"><code>{path_str}</code></div>""",
                        unsafe_allow_html=True,
                    )

                with col2:
                    # Show individual route map
                    individual_map = self.visualizer.create_route_map(
                        route["route_info"], route["traffic_level"]
                    )
                    folium_static(individual_map, width=1200)

                # Show route details in a table
                st.markdown("**Step-by-Step Navigation:**")
                route_df = pd.DataFrame(
                    [
                        {
                            "Step": j + 1,
                            "From Site": step["from_id"],
                            "To Site": step["to_id"],
                            "Road": step["road"],
                            "Distance (km)": f"{step['distance']:.2f}",
                            "Traffic Flow (veh/hr)": int(step["traffic_volume"]),
                            "Travel Time (min)": f"{step['travel_time']:.2f}",
                            "Time of Day": step["time_of_day"],
                        }
                        for j, step in enumerate(route["route_info"])
                    ]
                )
                st.dataframe(route_df, hide_index=True, use_container_width=True)

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
        search_method,
        k_value,
    ):
        """
        Find and display routes between origin and destination using selected search method
        """
        # Create a progress bar with better styling
        if search_method == "K-Shortest":
            progress_text = f"Finding {k_value} shortest paths using Yen's algorithm with {selected_model} model for {datetime_str}..."
        else:
            progress_text = f"Finding optimal routes using {selected_model} model for {datetime_str}..."

        progress_bar = st.progress(0)

        # Display a spinner with a better message
        with st.spinner(progress_text):
            # Simulate progress for better UX
            for i in range(100):
                # Update progress bar
                progress_bar.progress(i + 1)
                # Smaller sleep for faster response
                time_module.sleep(0.01)

            # Find routes using the selected search method
            if search_method == "K-Shortest":
                routes = self.route_finder.find_top_k_routes(
                    origin_id,
                    destination_id,
                    k=k_value,
                    prediction_model=selected_model,
                    datetime_str=datetime_str,
                )
            else:
                # Traditional algorithms
                routes = self.route_finder.find_multiple_routes(
                    origin_id,
                    destination_id,
                    selected_algorithms,
                    selected_model,
                    datetime_str,
                )

            # Check if routes were found
            if not routes:
                if search_method == "K-Shortest":
                    st.error(
                        f"No route found from Site {origin_id} to Site {destination_id} using Yen's algorithm. Try different sites."
                    )
                else:
                    st.error(
                        f"No route found from Site {origin_id} to Site {destination_id}. Try different sites or algorithms."
                    )
                return

            # Display routes
            self._display_route_results(
                origin_id, destination_id, routes, selected_model
            )

            # Clear the progress bar when done
            progress_bar.empty()

    def _render_implementation_notes(self):
        """
        Render implementation notes expander
        """
        with st.expander("Implementation Notes"):
            st.info(
                """
            **Current Implementation Notes:**
            
            - The route finder supports two approaches:
              1. **Yen's K-Shortest Paths Algorithm**: 
                 * Finds the k shortest paths from origin to destination
                 * Uses A* as the underlying search algorithm
                 * Excellent for finding alternative routes with similar efficiency
                 * Provides more diverse routing options

              2. **Traditional Search Algorithms**:
                 * A* (A-star): Optimal path finding using distance heuristic
                 * DFS (Depth-First Search): Explores as far as possible along each branch
                 * BFS (Breadth-First Search): Explores all neighbors at current depth before moving deeper
                 * GBFS (Greedy Best-First Search): Always moves toward the goal using heuristic
                 * UCS (Uniform Cost Search): Explores paths in order of increasing cost
            
            - Travel time is calculated based on distance and predicted traffic volumes
            - Different colors are used to distinguish between routes (aqua, blue, green, orange, red, etc.)
            - A 30-second delay is added for each intersection as per the assignment requirements
            - Three prediction models (LSTM, GRU, CNN-LSTM) are available for traffic prediction
            """
            )

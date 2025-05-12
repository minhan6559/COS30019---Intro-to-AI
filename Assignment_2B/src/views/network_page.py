import streamlit as st
from streamlit_folium import folium_static
from src.views.base_page import BasePage


class NetworkPage(BasePage):
    """
    Page for displaying the network map and connection details
    """

    def __init__(self, network, visualizer):
        """
        Initialize with a SiteNetwork object and a NetworkVisualizer
        """
        super().__init__(network)
        self.visualizer = visualizer

    def render(self):
        """
        Render the network map page
        """

        # Add an info box with network stats
        with st.container():
            st.markdown(
                """
            <div style='background-color: #f0f5ff; padding: 20px; border-radius: 10px; border: 1px solid #d0e0ff; margin-bottom: 20px;'>
                <h2 style='text-align: center; color: #0066CC; margin-top: 0;'>SCATS Network Explorer</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Network statistics in a nice card layout
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"""
                <div style='background-color: #eef7ff; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #d0e0ff;'>
                    <h3 style='margin-top: 0;'>🚦 Total Sites</h3>
                    <p style='font-size: 24px; font-weight: bold;'>{len(self.network.sites_data)}</p>
                    <p>SCATS Traffic Sites</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div style='background-color: #eef7ff; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #d0e0ff;'>
                    <h3 style='margin-top: 0;'>🔄 Total Connections</h3>
                    <p style='font-size: 24px; font-weight: bold;'>{len(self.network.connections)}</p>
                    <p>Directed Road Links</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                # Calculate average connections per site
                avg_connections = len(self.network.connections) / len(
                    self.network.sites_data
                )
                st.markdown(
                    f"""
                <div style='background-color: #eef7ff; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #d0e0ff;'>
                    <h3 style='margin-top: 0;'>🌐 Network Density</h3>
                    <p style='font-size: 24px; font-weight: bold;'>{avg_connections:.1f}</p>
                    <p>Avg. Connections per Site</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Create tabs for different views
        tab1, tab2 = st.tabs(["🗺️ Network Map", "🔍 Site Search"])

        with tab1:
            # Show the default map with all connections
            st.markdown("### Complete Network Visualization")
            st.caption(
                "This map shows all SCATS site locations and directed connections between them"
            )

            # Set custom CSS for the map container to use full width
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

            with st.spinner("Generating network map... This may take a moment."):
                m = self.visualizer.create_map()
                folium_static(m, width=1200)

            # Show connection details dataframe directly (without button)
            st.subheader("Complete Connections Database")
            with st.spinner("Loading connections database..."):
                conn_df = self.visualizer.create_connections_dataframe()
                st.dataframe(conn_df, hide_index=True, use_container_width=True)

        with tab2:
            st.markdown("### Search For Specific Site")
            st.caption("Select a site to view its connections and details")

            # Use a more prominent selection
            site_ids = sorted(list(map(int, self.network.sites_data.keys())))

            # Custom CSS to set fixed width for the select box
            st.markdown(
                """
                <style>
                [data-testid="stSelectbox"] {
                    max-width: 250px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Left-aligned site selector with smaller width
            selected_site = st.selectbox(
                "Select site ID:",
                options=site_ids,
                key="site_filter",
                format_func=lambda x: f"Site {x}",
            )

            if selected_site:
                # Display highlighted map
                with st.spinner("Generating map for selected site..."):
                    highlighted_map = self.visualizer.create_map(
                        highlighted_site=selected_site
                    )
                    folium_static(highlighted_map, width=1200)

                # Get site details
                site_data = self.network.get_site(selected_site)

                # Create a nice info card for the site
                st.markdown(
                    f"""
                <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 15px 0;'>
                    <h3 style='margin-top: 0;'>Site {selected_site} Details</h3>
                    <p><strong>Latitude:</strong> {site_data['latitude']}</p>
                    <p><strong>Longitude:</strong> {site_data['longitude']}</p>
                    <p><strong>Connected Roads:</strong> {', '.join(site_data['connected_roads'])}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Get outgoing and incoming connections
                outgoing = self.network.get_outgoing_connections(selected_site)
                incoming = self.network.get_incoming_connections(selected_site)

                # Create statistics row
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                    <div style='background-color: #e6f7ff; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #d0e0ff;'>
                        <h4 style='margin-top: 0;'>Outgoing Connections</h4>
                        <p style='font-size: 24px; font-weight: bold;'>{len(outgoing)}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                    <div style='background-color: #e6f7ff; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #d0e0ff;'>
                        <h4 style='margin-top: 0;'>Incoming Connections</h4>
                        <p style='font-size: 24px; font-weight: bold;'>{len(incoming)}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Show connections in tabs
                conn_tabs = st.tabs(["Outgoing Connections", "Incoming Connections"])

                with conn_tabs[0]:
                    # Show outgoing connections
                    if outgoing:
                        out_df, _ = self.visualizer.create_filtered_dataframes(
                            selected_site
                        )
                        if out_df is not None:
                            st.dataframe(
                                out_df, hide_index=True, use_container_width=True
                            )
                    else:
                        st.info("No outgoing connections from this site")

                with conn_tabs[1]:
                    # Show incoming connections
                    if incoming:
                        _, in_df = self.visualizer.create_filtered_dataframes(
                            selected_site
                        )
                        if in_df is not None:
                            st.dataframe(
                                in_df, hide_index=True, use_container_width=True
                            )
                    else:
                        st.info("No incoming connections to this site")

    def _render_connection_details(self):
        """
        Render the connection details section
        """
        if st.checkbox("Show all Connection Details"):
            conn_df = self.visualizer.create_connections_dataframe()
            st.dataframe(conn_df, hide_index=True)

    def _render_connection_filter(self):
        """
        Render the connection filtering section
        """
        if st.checkbox("Search for Site"):
            site_ids = sorted(list(map(int, self.network.sites_data.keys())))
            # Hide the selectbox label via CSS
            hide_label_style = """
                <style>
                div[data-baseweb="select"] > div:first-child {
                    display: none;
                }
                </style>
            """
            st.markdown(hide_label_style, unsafe_allow_html=True)

            filter_site = st.selectbox(
                "Select site ID to find connections", options=site_ids
            )

            # Get outgoing and incoming connections
            outgoing = self.network.get_outgoing_connections(filter_site)
            incoming = self.network.get_incoming_connections(filter_site)

            st.write(
                f"Site {filter_site} has {len(outgoing)} outgoing and {len(incoming)} incoming connections"
            )

            # Generate dataframes
            out_df, in_df = self.visualizer.create_filtered_dataframes(filter_site)

            # Show outgoing connections
            st.subheader("Outgoing Connections")
            if out_df is not None:
                st.dataframe(out_df, hide_index=True)
            else:
                st.write("No outgoing connections")

            # Show incoming connections
            st.subheader("Incoming Connections")
            if in_df is not None:
                st.dataframe(in_df, hide_index=True)
            else:
                st.write("No incoming connections")

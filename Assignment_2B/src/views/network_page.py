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
        # Set page title
        st.title("Traffic-Based Route Guidance System (TBRGS)")
        st.header("SCATS Sites Directed Network Map")

        # Show connection statistics
        # st.write(f"Total SCATS sites: {len(self.network.sites_data)}")
        # st.write(f"Total directed connections: {len(self.network.connections)}")

        # Replace checkbox columns with radio button
        display_option = st.radio(
            "Select display option:",
            options=["Show all Connection Details", "Search for Site"],
            horizontal=True,
        )

        if display_option == "Show all Connection Details":
            # Show the default map with all connections
            with st.spinner(
                "Generating directed site connections... This may take a moment."
            ):
                m = self.visualizer.create_map()
                folium_static(m)

            # Show connection details dataframe
            conn_df = self.visualizer.create_connections_dataframe()
            st.dataframe(conn_df, hide_index=True)

        elif display_option == "Search for Site":
            site_ids = sorted(list(map(int, self.network.sites_data.keys())))
            selected_site = st.selectbox(
                "Select site ID to find connections",
                options=site_ids,
                key="site_filter",
            )

            # Display highlighted map
            with st.spinner("Generating map with selected site..."):
                highlighted_map = self.visualizer.create_map(
                    highlighted_site=selected_site
                )
                folium_static(highlighted_map)

            # Get outgoing and incoming connections
            outgoing = self.network.get_outgoing_connections(selected_site)
            incoming = self.network.get_incoming_connections(selected_site)

            st.write(
                f"Site {selected_site} has {len(outgoing)} outgoing and {len(incoming)} incoming connections"
            )

            # Generate dataframes
            out_df, in_df = self.visualizer.create_filtered_dataframes(selected_site)

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

import streamlit as st
from src.route_finder.site_network import SiteNetwork
from src.route_finder.route_finder import RouteFinder
from src.visualizer.base_visualizer import BaseVisualizer
from src.visualizer.network_visualizer import NetworkVisualizer
from src.visualizer.route_visualizer import RouteVisualizer
from src.views.network_page import NetworkPage
from src.views.route_page import RoutePage


class TBRGSApp:
    """
    Main Streamlit application class for the Traffic-Based Route Guidance System
    """

    # In app.py
    def __init__(self, metadata_file="data/sites_metadata.json"):
        """
        Initialize the application
        """
        # Initialize core components
        self.network = SiteNetwork(metadata_file)
        self.route_finder = RouteFinder(self.network)

        # Initialize UI components with specialized visualizers
        self.network_visualizer = NetworkVisualizer(self.network)
        self.route_visualizer = RouteVisualizer(self.network)

        # Initialize pages
        self.network_page = NetworkPage(self.network, self.network_visualizer)
        self.route_page = RoutePage(
            self.network, self.route_finder, self.route_visualizer
        )

    def run(self):
        """
        Run the Streamlit application
        """
        # Create a sidebar for navigation
        st.sidebar.title("TBRGS Navigation")
        page = st.sidebar.radio("Select a page:", ["Network Map", "Route Finder"])

        # Display the selected page
        if page == "Network Map":
            self.network_page.render()
        elif page == "Route Finder":
            self.route_page.render()


# Run the application if this is the main script
if __name__ == "__main__":
    app = TBRGSApp("processed_data/preprocessed_data/sites_metadata.json")
    app.run()

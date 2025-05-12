import streamlit as st
import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Store the Assignment_2B directory path
assignment_dir = os.path.join(current_dir, "Assignment_2B")

# Add paths relative to the script location
sys.path.append(assignment_dir)

# Change working directory to Assignment_2B for relative imports
# This way relative paths will work as expected
os.chdir(assignment_dir)

from src.route_finder.site_network import SiteNetwork
from src.route_finder.route_finder import RouteFinder
from src.visualizer.network_visualizer import NetworkVisualizer
from src.visualizer.route_visualizer import RouteVisualizer
from src.views.network_page import NetworkPage
from src.views.route_page import RoutePage
from streamlit_app import TBRGSApp


# Run the application if this is the main script
if __name__ == "__main__":
    app = TBRGSApp("processed_data/preprocessed_data/sites_metadata.json")
    app.run()

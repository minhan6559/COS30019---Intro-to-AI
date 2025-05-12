import streamlit as st
from PIL import Image
from src.route_finder.site_network import SiteNetwork
from src.route_finder.route_finder import RouteFinder
from src.visualizer.network_visualizer import NetworkVisualizer
from src.visualizer.route_visualizer import RouteVisualizer
from src.views.network_page import NetworkPage
from src.views.route_page import RoutePage


class TBRGSApp:
    """
    Main Streamlit application class for the Traffic-Based Route Guidance System
    """

    def __init__(self, metadata_file="data/sites_metadata.json"):
        """
        Initialize the application
        """
        # Set page configuration
        st.set_page_config(
            page_title="TBRGS - Traffic-Based Route Guidance System",
            page_icon="🚦",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Apply custom CSS
        self._apply_custom_css()

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

    def _apply_custom_css(self):
        """
        Apply custom CSS to improve the UI appearance
        """
        st.markdown(
            """
        <style>
        /* Make the sidebar look nicer */
        section[data-testid="stSidebar"] {
            background-color: #f0f2f6;
            border-right: 1px solid #ddd;
        }
        
        /* Improve button styling */
        .stButton>button {
            width: 100%;
            border-radius: 4px;
            font-weight: 500;
            padding: 0.5rem 1rem;
            transition: all 0.2s;
        }
        
        /* Improve header styling */
        h1 {
            color: #0066CC;
            padding-bottom: 1rem;
            border-bottom: 1px solid #eee;
        }
        
        h2 {
            color: #2070b0;
            margin-top: 1rem;
        }
        
        /* Make the dataframes more compact */
        .dataframe {
            font-size: 0.9rem !important;
        }
        
        /* Add padding to expanders */
        .streamlit-expanderContent {
            padding: 1rem;
            background-color: #f9f9f9;
            border-radius: 0 0 4px 4px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def run(self):
        """
        Run the Streamlit application
        """
        # Create a sidebar for navigation
        with st.sidebar:
            logo = Image.open("images/swinburne_logo.png")
            st.image(
                logo,
            )
            st.title("Boroondara TBRGS Navigation")

            # Add some space
            st.markdown("---")

            # Navigation options with icons
            page = st.radio(
                "Select a page:", options=["🗺️ Network Map", "🚗 Route Finder"]
            )

            # Add some descriptive text
            if "Network Map" in page:
                st.info(
                    "View the complete network of SCATS sites and their connections."
                )
            else:
                st.info(
                    "Find optimal routes between sites based on traffic conditions."
                )

            st.markdown("---")
            st.caption("© Created by: Minh An Nguyen")

        # Display the selected page
        if "Network Map" in page:
            self.network_page.render()
        elif "Route Finder" in page:
            self.route_page.render()


# Run the application if this is the main script
if __name__ == "__main__":
    app = TBRGSApp("processed_data/preprocessed_data/sites_metadata.json")
    app.run()

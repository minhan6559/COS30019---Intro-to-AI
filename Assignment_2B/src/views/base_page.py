import streamlit as st

class BasePage:
    """
    Base class for all UI pages in the TBRGS application
    """
    def __init__(self, network):
        """
        Initialize with a SiteNetwork object
        """
        self.network = network
    
    def render(self):
        """
        Render the page - must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement render()")
    
    def display_site_info(self, site_id):
        """
        Common method to display site information
        """
        site_data = self.network.get_site(site_id)
        if site_data:
            st.write(f"Connected roads: {', '.join(site_data['connected_roads'])}")
            #st.write(f"Locations: {', '.join(site_data['locations'])}")
    
    def get_road_intersection(self, site_id):
        """
        Get a simplified description of the site's location based on connected roads
        """
        site_data = self.network.get_site(site_id)
        if site_data and len(site_data['connected_roads']) >= 2:
            roads = site_data['connected_roads'][:2]  # Take first two roads
            return f"{roads[0]}/{roads[1]}"
        elif site_data and len(site_data['connected_roads']) == 1:
            return site_data['connected_roads'][0]
        else:
            return "Unknown location"
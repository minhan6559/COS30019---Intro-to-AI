import pandas as pd
import folium
from folium.plugins import AntPath

class BaseVisualizer:
    """
    Base class for map visualization with common functionality
    """
    def __init__(self, network):
        """
        Initialize with a SiteNetwork object
        """
        self.network = network
    
    def _create_base_map(self, center=None, zoom_start=12):
        """
        Create a base folium map with the specified center and zoom level
        """
        if center is None:
            # Calculate map center from all sites
            center = [
                self.network.sites_df['latitude'].mean(), 
                self.network.sites_df['longitude'].mean()
            ]
        
        # Create the map
        return folium.Map(location=center, zoom_start=zoom_start)
    
    def _add_all_sites(self, m, color='blue', radius=8, opacity=0.7):
        """
        Add all sites to the map with specified style
        """
        for site_id, site in self.network.sites_data.items():
            folium.CircleMarker(
                location=[site['latitude'], site['longitude']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=opacity,
                popup=f"Site ID: {site_id}<br>Roads: {', '.join(site['connected_roads'])}"
            ).add_to(m)
        
        return m
    
    def _add_background_sites(self, m, radius=5, color='gray', opacity=0.5):
        """
        Add all sites as background markers (typically used for route maps)
        """
        for site_id, site in self.network.sites_data.items():
            folium.CircleMarker(
                location=[site['latitude'], site['longitude']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=opacity,
                popup=f"Site ID: {site_id}<br>Roads: {', '.join(site['connected_roads'])}"
            ).add_to(m)
        
        return m
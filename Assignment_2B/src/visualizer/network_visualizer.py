import pandas as pd
import folium
from folium.plugins import AntPath
from src.visualizer.base_visualizer import BaseVisualizer


class NetworkVisualizer(BaseVisualizer):
    """
    Specialized class for visualizing the network of SCATS sites and their connections
    """

    def create_map(self, highlighted_site=None):
        """
        Create a folium map with all sites and connections
        """
        # Create the base map
        m = self._create_base_map()

        # Add sites to the map
        for site_id, site in self.network.sites_data.items():
            if highlighted_site is not None and int(site_id) == highlighted_site:
                # Highlight the selected site
                folium.CircleMarker(
                    location=[site["latitude"], site["longitude"]],
                    radius=12,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.9,
                    popup=f"<b>SELECTED SITE</b><br>Site ID: {site_id}<br>Roads: {', '.join(site['connected_roads'])}",
                    tooltip=f"SELECTED: Site {site_id}",
                ).add_to(m)
            else:
                # Regular sites
                folium.CircleMarker(
                    location=[site["latitude"], site["longitude"]],
                    radius=8,
                    color="blue",
                    fill=True,
                    fill_color="blue",
                    fill_opacity=0.7,
                    popup=f"Site ID: {site_id}<br>Roads: {', '.join(site['connected_roads'])}",
                ).add_to(m)

        # Add connections to the map
        self._add_connections_to_map(m, highlighted_site)

        return m

    def _add_connections_to_map(self, m, highlighted_site=None):
        """
        Add all connections to the map
        """
        for conn in self.network.connections:
            # Create arrow using folium's plugins
            locations = [
                [conn["from_lat"], conn["from_lng"]],
                [conn["to_lat"], conn["to_lng"]],
            ]

            # Add directional arrows
            arrow = folium.PolyLine(
                locations=locations,
                color="red",
                weight=2,
                opacity=0.6,
                arrow_style="3",
                arrow_color="red",
                show_arrow=True,
            )

            arrow.add_to(m)

    def create_connections_dataframe(self):
        """
        Create a DataFrame with all connection details
        """
        return pd.DataFrame(
            [
                {
                    "From Site": conn["from_id"],
                    "To Site": conn["to_id"],
                    "Shared Road": conn["shared_road"],
                    "Approach Location": conn["approach_location"],
                    "Distance (km)": round(conn["distance"], 2),
                }
                for conn in self.network.connections
            ]
        )

    def create_filtered_dataframes(self, site_id):
        """
        Create DataFrames for outgoing and incoming connections for a specific site
        """
        outgoing = self.network.get_outgoing_connections(site_id)
        incoming = self.network.get_incoming_connections(site_id)

        out_df = None
        in_df = None

        if outgoing:
            out_df = pd.DataFrame(
                [
                    {
                        "To Site": c["to_id"],
                        "Shared Road": c["shared_road"],
                        "Approach Location": c["approach_location"],
                        "Distance (km)": round(c["distance"], 2),
                    }
                    for c in outgoing
                ]
            )

        if incoming:
            in_df = pd.DataFrame(
                [
                    {
                        "From Site": c["from_id"],
                        "Shared Road": c["shared_road"],
                        "Approach Location": c["approach_location"],
                        "Distance (km)": round(c["distance"], 2),
                    }
                    for c in incoming
                ]
            )

        return out_df, in_df

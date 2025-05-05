import json
import pandas as pd
from haversine import haversine, Unit


class SiteNetwork:
    """
    Class for representing the network of SCATS sites and their connections
    """

    def __init__(self, metadata_file="data/sites_metadata.json"):
        """
        Initialize the network with data from a metadata file
        """
        # Load SCATS site metadata
        with open(metadata_file, "r") as f:
            self.sites_data = json.load(f)

        # Apply coordinate correction to align with actual intersections
        self._adjust_coordinates()

        # Convert to DataFrame for easier manipulation
        self.sites_df = pd.DataFrame(
            [
                {
                    "site_id": int(site_id),
                    "latitude": data["latitude"],
                    "longitude": data["longitude"],
                    "connected_roads": data["connected_roads"],
                    "locations": data["locations"],
                }
                for site_id, data in self.sites_data.items()
            ]
        )

        # Generate connections between sites
        self.connections = self.find_directed_connections()

    def _adjust_coordinates(self):
        """
        Apply a correction to latitude and longitude to align with actual intersections
        Based on observation that sites need to be shifted slightly northeast
        """
        # Correction factors for Melbourne area
        lat_correction = 0.0012  # Shift north
        lng_correction = 0.0012  # Shift east

        # Apply the correction to all sites
        for site_id in self.sites_data:
            self.sites_data[site_id]["latitude"] += lat_correction
            self.sites_data[site_id]["longitude"] += lng_correction

    def find_directed_connections(self, max_distance=5.0):
        """
        Find directed connections between sites based on shared roads and approach directions,
        only connecting sites that are truly adjacent on a road
        """

        """
        If Site 101 and Site 205 are both on WARRIGAL_RD, we store: road_to_sites["WARRIGAL_RD"] = [101, 205]
        """
        connections = []
        road_to_sites = {}

        # Group sites by road
        for site_id, site in self.sites_data.items():
            for road in site["connected_roads"]:
                if road not in road_to_sites:
                    road_to_sites[road] = []
                road_to_sites[road].append(site_id)

        # Create connections for each road
        for road, site_ids in road_to_sites.items():
            self._process_road_connections(road, site_ids, max_distance, connections)

        return connections

    def _process_road_connections(self, road, site_ids, max_distance, connections):
        """
        Process connections for a specific road
        """
        # Skip roads with only one site
        if len(site_ids) <= 1:
            return

        # Get site data for all sites on this road
        sites_on_road = []
        for site_id in site_ids:
            site_data = self.sites_data[site_id]
            # Skip sites with invalid coordinates
            if site_data["latitude"] == 0 and site_data["longitude"] == 0:
                continue
            sites_on_road.append(
                {
                    "id": site_id,
                    "latitude": site_data["latitude"],
                    "longitude": site_data["longitude"],
                    "data": site_data,
                }
            )

        # Skip if we don't have enough sites after filtering
        if len(sites_on_road) <= 1:
            return

        # Determine road orientation
        is_east_west = self._determine_road_orientation(sites_on_road)

        # Sort sites by their position along the road
        if is_east_west:
            sites_on_road.sort(key=lambda x: x["longitude"])
        else:
            sites_on_road.sort(key=lambda x: x["latitude"])

        # Create connections between adjacent sites
        for i in range(len(sites_on_road) - 1):
            site1 = sites_on_road[i]
            site2 = sites_on_road[i + 1]

            # Calculate distance between sites
            start = (site1["latitude"], site1["longitude"])
            end = (site2["latitude"], site2["longitude"])
            distance = haversine(start, end, unit=Unit.KILOMETERS)

            # Only consider connections if they're within reasonable distance
            if distance < max_distance:
                self._create_directional_connections(
                    site1, site2, road, distance, connections
                )

    def _determine_road_orientation(self, sites_on_road):
        """
        Determine if a road is east-west or north-south based on coordinate variance
        """
        lat_values = [site["latitude"] for site in sites_on_road]
        lng_values = [site["longitude"] for site in sites_on_road]
        lat_variance = sum(
            (lat - sum(lat_values) / len(lat_values)) ** 2 for lat in lat_values
        )
        lng_variance = sum(
            (lng - sum(lng_values) / len(lng_values)) ** 2 for lng in lng_values
        )

        # If longitude varies more than latitude, it's likely an east-west road
        return lng_variance > lat_variance

    def _create_directional_connections(
        self, site1, site2, road, distance, connections
    ):
        """
        Create directional connections between two sites if approach locations match
        """
        # Calculate geographic direction
        lat_diff = site2["latitude"] - site1["latitude"]
        lng_diff = site2["longitude"] - site1["longitude"]

        # Determine the cardinal direction
        if abs(lat_diff) > abs(lng_diff):
            # North-south movement
            cardinal_direction = "N" if lat_diff > 0 else "S"
        else:
            # East-west movement
            cardinal_direction = "E" if lng_diff > 0 else "W"

        # Create forward and backward connections
        opposite_direction = {"N": "S", "S": "N", "E": "W", "W": "E"}

        # Forward connection (site1 to site2)
        approach_dir = opposite_direction.get(cardinal_direction, "")
        self._try_create_connection(
            site1, site2, road, distance, approach_dir, connections
        )

        # Backward connection (site2 to site1)
        backward_dir = cardinal_direction
        self._try_create_connection(
            site2, site1, road, distance, backward_dir, connections
        )

    def _try_create_connection(
        self, from_site, to_site, road, distance, approach_dir, connections
    ):
        """
        Try to create a connection if approach location matches
        """
        # Look for matching approach location
        location = None
        for loc in to_site["data"]["locations"]:
            if (
                road.lower() in loc.lower()
                and f"{approach_dir} of".lower() in loc.lower()
            ):
                location = loc
                break

        # Create connection if approach location found
        if location:
            connections.append(
                {
                    "from_id": int(from_site["id"]),
                    "to_id": int(to_site["id"]),
                    "shared_road": road,
                    "from_lat": from_site["latitude"],
                    "from_lng": from_site["longitude"],
                    "to_lat": to_site["latitude"],
                    "to_lng": to_site["longitude"],
                    "distance": distance,
                    "approach_location": location,
                }
            )

    def get_site(self, site_id):
        """
        Get site data by ID
        """
        return self.sites_data.get(str(site_id))

    def get_outgoing_connections(self, site_id):
        """
        Get all connections originating from a specific site
        """
        return [c for c in self.connections if c["from_id"] == site_id]

    def get_incoming_connections(self, site_id):
        """
        Get all connections arriving at a specific site
        """
        return [c for c in self.connections if c["to_id"] == site_id]


if __name__ == "__main__":
    # Example usage
    network = SiteNetwork("processed_data\preprocessed_data\sites_metadata.json")
    print(network.connections)

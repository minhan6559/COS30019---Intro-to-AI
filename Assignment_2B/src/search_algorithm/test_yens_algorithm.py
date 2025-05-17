import os
import sys
import datetime
import inspect

# Add the project root directory to path to make imports work
sys.path.insert(0, os.getcwd())

from src.route_finder.route_finder import RouteFinder
from src.route_finder.site_network import SiteNetwork


def main():
    print("Loading site network...")
    network = SiteNetwork()

    print("Creating route finder...")
    route_finder = RouteFinder(network)

    # Parameters for testing
    origin_id = 970
    destination_id = 4263
    model = "LSTM"
    datetime_str = "2006-10-01 08:00"
    k = 5

    print(f"Finding top {k} routes from {origin_id} to {destination_id}...")
    print(f"Model: {model}, DateTime: {datetime_str}")

    # Find routes using Yen's algorithm
    routes = route_finder.find_top_k_routes(
        origin_id,
        destination_id,
        k=k,
        prediction_model=model,
        datetime_str=datetime_str,
    )

    # Display results
    print(f"\nFound {len(routes)} routes:")
    for i, route in enumerate(routes):
        print(f"\nRoute {i+1} ({route['algorithm']}):")
        print(f"  Rank: {route['route_rank']}")
        print(f"  Travel Time: {route['total_cost']:.2f} minutes")
        print(f"  Path: {' -> '.join(map(str, route['path']))}")
        print(f"  Traffic Level: {route['traffic_level']}")

        if route["route_info"]:
            print("\n  Detailed Steps:")
            for step in route["route_info"]:
                print(f"    {step['from_id']} -> {step['to_id']} ({step['road']})")
                print(f"      Distance: {step['distance']:.2f} km")
                print(f"      Travel Time: {step['travel_time']:.2f} minutes")
                print(f"      Traffic Volume: {step['traffic_volume']}")
                print(f"      Time of Day: {step['time_of_day']}")


if __name__ == "__main__":
    main()

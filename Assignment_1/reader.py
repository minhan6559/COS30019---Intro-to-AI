from src.parser.graph_parser import GraphParser, create_graph_problem_from_file
import os


def main():
    """Example usage of the GraphParser and helper functions"""

    # You can now use different ways to specify the file path:

    # 1. Just the filename (will be searched in multiple locations)
    filename = "PathFinder-test.txt"

    # 2. Absolute path
    # filename = os.path.abspath("PathFinder-test.txt")

    # 3. Path relative to testcases directory
    # filename = os.path.join("testcases", "PathFinder-test.txt")

    print(f"Attempting to parse graph file: {filename}")

    # Using the OOP approach
    print("\nUsing GraphParser class:")
    print("------------------------")
    parser = GraphParser()
    parser.parse_file(filename)
    graph = parser.create_graph()

    print("Parsed Graph Data:")
    print("------------------")
    print("Nodes (with coordinates):")
    for node, coords in parser.locations.items():
        print(f"{node}: {coords}")

    print("\nGraph Structure:")
    for src, neighbors in graph.graph_dict.items():
        print(f"{src} -> {neighbors}")

    print("\nOrigin:", parser.origin)
    print("Destinations:", parser.destinations)

    # Using the helper function
    print("\nUsing helper function:")
    print("----------------------")
    graph, origin, destinations, locations = create_graph_problem_from_file(filename)

    print("Locations:", locations)
    print("Origin:", origin)
    print("Destinations:", destinations)

    return 0


if __name__ == "__main__":
    main()

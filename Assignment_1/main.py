def main():
    """Example usage of the GraphParser"""
    # Use relative path from script location
    test_file = os.path.join("testcases", "testcase1.txt")

    print("Using GraphParser class:")
    print("------------------------")
    try:
        parser = GraphParser()
        parser.parse_file(test_file)
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

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the testcases directory exists and contains testcase1.txt")
        return 1
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()

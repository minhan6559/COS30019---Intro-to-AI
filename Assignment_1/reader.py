def parse_graph_file(filename):
    """
    Parses a file containing graph data in the following format:
    
    Nodes:
    1: (4,1)
    2: (2,2)
    ...
    
    Edges:
    (2,1): 4
    (3,1): 5
    ...
    
    Origin:
    2
    
    Destinations:
    5; 4
    
    Returns:
        nodes: A dictionary mapping node id to (x, y) coordinates.
        graph: An adjacency list (dictionary) mapping each node to a list of (neighbor, weight) tuples.
        origin: The starting node.
        destinations: A list of destination nodes.
    """
    nodes = {}
    graph = {}
    origin = None
    destinations = []
    
    section = None
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # Identify the section headers
            if line.startswith("Nodes:"):
                section = "nodes"
                continue
            elif line.startswith("Edges:"):
                section = "edges"
                continue
            elif line.startswith("Origin:"):
                section = "origin"
                continue
            elif line.startswith("Destinations:"):
                section = "destinations"
                continue

            # Process each section accordingly
            if section == "nodes":
                # Expected format: 1: (4,1)
                try:
                    node_id_str, coord_str = line.split(":")
                    node_id = int(node_id_str.strip())
                    coord_str = coord_str.strip().strip("()")
                    x_str, y_str = coord_str.split(",")
                    nodes[node_id] = (int(x_str.strip()),int(y_str.strip()))
                except Exception as e:
                    print(f"Error parsing node line: {line} -> {e}")

            elif section == "edges":
                # Expected fo
                # rmat: (2,1): 4
                try:
                    left, weight_str = line.split(":")
                    weight = int(weight_str.strip())
                    left = left.strip().strip("()")
                    src_str, dest_str = left.split(",")
                    src = int(src_str.strip())
                    dest = int(dest_str.strip())
                    
                    # Add the edge to the graph (directed edge)
                    if src not in graph:
                        graph[src] = []
                    graph[src].append((dest, weight))
                except Exception as e:
                    print(f"Error parsing edge line: {line} -> {e}")

            elif section == "origin":
                # The origin should be a single number
                try:
                    origin = int(line)
                except Exception as e:
                    print(f"Error parsing origin: {line} -> {e}")

            elif section == "destinations":
                # Destinations might be separated by semicolons
                for dest in line.split(";"):
                    dest = dest.strip()
                    if dest:
                        try:
                            destinations.append(int(dest))
                        except Exception as e:
                            print(f"Error parsing destination: {dest} -> {e}")

    return nodes, graph, origin, destinations


# def main():
#     filename = "PathFinder-test.txt"
#     nodes, graph, origin, destinations = parse_graph_file(filename)
    
#     print("Parsed Graph Data:")
#     print("------------------")
#     print("Nodes (with coordinates):")
#     for node, coords in nodes.items():
#         print(f"{node}:{coords}")
    
#     print("\nGraph (Adjacency List):")
#     for src, neighbours in graph.items():
#         print(f"{src} -> {neighbours}")
    
#     print("\nOrigin:", origin)
#     print("Destinations:", destinations)
    
#     return 0


# if __name__ == '__main__':
#     main()

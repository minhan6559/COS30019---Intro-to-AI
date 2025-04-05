import os

from src.graph.graph import Graph


class GraphParser:
    """
    Parser class for reading graph data from text files and creating Graph objects
    compatible with the search algorithms.
    """

    def __init__(self):
        """Initialize the parser with empty containers"""
        self.locations = {}
        self.graph_dict = {}
        self.origin = None
        self.destinations = []
        # Get the project root directory (2 levels up from the parser module)
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../..")
        )

    def parse_file(self, filename):
        """
        Parses a file containing graph data.

        Args:
            filename: Path to the file (absolute or relative to project root)

        Returns:
            self: The parser instance with populated data

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        # Try multiple path resolution strategies
        file_path = self._resolve_file_path(filename)

        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Graph file not found: {filename}")

        section = None

        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                # Skip empty lines or comment lines that start with #
                if not line or line.startswith("#"):
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
                    self._parse_node_line(line)
                elif section == "edges":
                    self._parse_edge_line(line)
                elif section == "origin":
                    self._parse_origin_line(line)
                elif section == "destinations":
                    self._parse_destinations_line(line)

        return self

    def _resolve_file_path(self, filename):
        """
        Resolve a file path using multiple strategies.

        Args:
            filename: The filename or path to resolve

        Returns:
            str: The resolved absolute file path, or None if not found
        """
        # Strategy 1: Check if it's an absolute path
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename

        # List of possible base directories to try
        base_dirs = [
            # Strategy 2: Relative to the current working directory
            os.getcwd(),
            # Strategy 3: Relative to the project root
            self.project_root,
            # Strategy 4: Relative to a 'testcases' directory in the project root
            os.path.join(self.project_root, "testcases"),
            # Strategy 5: Relative to the parser module directory
            os.path.dirname(os.path.abspath(__file__)),
        ]

        # Try each base directory
        for base_dir in base_dirs:
            potential_path = os.path.join(base_dir, filename)
            if os.path.exists(potential_path):
                return potential_path

        # If we get here, we couldn't find the file
        return None

    def _parse_node_line(self, line):
        """Parse a node line in the format '1: (4,1)'"""
        try:
            node_id_str, coord_str = line.split(":")
            node_id = int(node_id_str.strip())
            coord_str = coord_str.strip().strip("()")
            x_str, y_str = coord_str.split(",")
            self.locations[node_id] = (int(x_str.strip()), int(y_str.strip()))
        except Exception as e:
            print(f"Error parsing node line: {line} -> {e}")

    def _parse_edge_line(self, line):
        """Parse an edge line in the format '(2,1): 4'"""
        try:
            left, weight_str = line.split(":")
            weight = float(weight_str.strip())
            left = left.strip().strip("()")
            src_str, dest_str = left.split(",")
            src = int(src_str.strip())
            dest = int(dest_str.strip())

            # Initialize the source node's connections if needed
            if src not in self.graph_dict:
                self.graph_dict[src] = {}

            # Add the edge with its weight to match Graph class format
            self.graph_dict[src][dest] = weight

        except Exception as e:
            print(f"Error parsing edge line: {line} -> {e}")

    def _parse_origin_line(self, line):
        """Parse the origin line with a single integer"""
        try:
            self.origin = int(line)
        except Exception as e:
            print(f"Error parsing origin: {line} -> {e}")

    def _parse_destinations_line(self, line):
        """Parse destinations separated by semicolons"""
        for dest in line.split(";"):
            dest = dest.strip()
            if dest:
                try:
                    self.destinations.append(int(dest))
                except Exception as e:
                    print(f"Error parsing destination: {dest} -> {e}")

    def create_graph(self):
        """
        Create a Graph object from the parsed data

        Returns:
            graph: A Graph object with the parsed data and locations attached
        """
        graph = Graph(self.graph_dict)
        return graph

    def get_problem_components(self):
        """
        Returns all components needed to create a GraphProblem

        Returns:
            graph: A Graph object
            origin: The origin node
            destinations: List of destination nodes
            locations: Dictionary mapping nodes to their coordinates
        """
        graph = self.create_graph()
        return graph, self.origin, self.destinations, self.locations

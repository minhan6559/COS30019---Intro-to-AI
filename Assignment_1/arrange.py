import os
import shutil
from src.parser.graph_parser import GraphParser
from src.problem.multigoal_graph_problem import MultigoalGraphProblem
from src.graph.graph import Graph


def rearrange_files():
    """
    Load graphs from testcases folder, transform them, and save to testcases_flip folder.
    Transformations:
    1. Flip coordinates horizontally and vertically
    2. Add 1 to every node ID
    3. Add 1 to every path cost
    """
    # Define source and destination directories
    testcases_dir = os.path.join(os.getcwd(), "testcases")
    output_dir = os.path.join(os.getcwd(), "testcases_rearranged")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all .txt files in the testcases directory
    graph_files = sorted([f for f in os.listdir(testcases_dir) if f.endswith(".txt")])

    for i, index in enumerate([5, 7, 8, 6, 0, 3, 1, 4, 2]):
        original_file = graph_files[index]
        new_file = f"testcase_{i + 1}.txt"
        file_path = os.path.join(testcases_dir, original_file)
        new_file_path = os.path.join(output_dir, new_file)

        # Make a copy of the original file to the new file path
        shutil.copy(file_path, new_file_path)


if __name__ == "__main__":
    rearrange_files()

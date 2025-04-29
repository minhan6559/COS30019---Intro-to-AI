import os
import subprocess
import sys


def run_search_on_all_testcases(output_file="2_solution.txt"):
    # Path to the search.py script
    search_script = "search.py"

    # Path to the testcases folder
    testcases_dir = os.path.join("testcases_baovo")

    # Available search methods
    search_methods = ["BFS", "DFS", "AS", "GBFS", "CUS1", "CUS2"]

    # Check if testcases directory exists
    if not os.path.isdir(testcases_dir):
        print(f"Error: Testcases directory '{testcases_dir}' not found")
        return 1

    # Get all test case files (*.txt files excluding readme.txt)
    test_files = [
        f
        for f in os.listdir(testcases_dir)
        if f.endswith(".txt") and f.lower() != "readme.txt"
    ]

    if not test_files:
        print(f"No test files found in '{testcases_dir}'")
        return 1

    print(f"Found {len(test_files)} test files")
    print(f"Running {len(search_methods)} search algorithms on each test file")
    print(f"Results will be saved to {output_file}")

    # Clear output file if it exists
    with open(output_file, "w") as f:
        f.write(f"# Search Algorithm Test Results\n")
        f.write(f"# Generated on: {os.popen('date /t').read().strip()}\n\n")

    # Counter for tracking progress
    total_tests = len(test_files) * len(search_methods)
    completed_tests = 0

    # Run each search method on each test file
    for test_file in sorted(test_files):
        test_file_path = os.path.join(testcases_dir, test_file)

        for method in search_methods:
            completed_tests += 1
            print(
                f"Progress: {completed_tests}/{total_tests} - Running {method} on {test_file}",
                end="\r",
            )

            # Run the search.py script with the current test file and method
            result = subprocess.run(
                [sys.executable, search_script, test_file_path, method],
                capture_output=True,
                text=True,
            )

            # Write the output to the solution file
            with open(output_file, "a") as f:
                f.write(f"## Test: {test_file} - Method: {method}\n")
                f.write(result.stdout)
                f.write("\n" + "-" * 80 + "\n\n")

    print(f"\nCompleted all tests. Results saved to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(run_search_on_all_testcases())

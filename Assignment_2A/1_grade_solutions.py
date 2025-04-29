import os
import re
import subprocess
import sys


def run_search_on_all_testcases(output_file="3_student_output.txt", timeout=10):
    """Run all search algorithms on all test cases and save output to the specified file"""
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
    print(f"Timeout set to {timeout} seconds per test")
    print(f"Results will be saved to {output_file}")

    # Clear output file if it exists
    with open(output_file, "w") as f:
        f.write(f"# Search Algorithm Test Results\n")
        f.write(f"# Generated on: {os.popen('date /t').read().strip()}\n")
        f.write(f"# Timeout set to {timeout} seconds\n\n")

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

            try:
                # Run the search.py script with the current test file and method with timeout
                result = subprocess.run(
                    [sys.executable, search_script, test_file_path, method],
                    capture_output=True,
                    text=True,
                    timeout=timeout,  # Add timeout parameter
                )

                # Write the output to the solution file
                with open(output_file, "a") as f:
                    f.write(f"## Test: {test_file} - Method: {method}\n")
                    f.write(result.stdout)
                    f.write("\n" + "-" * 80 + "\n\n")

            except subprocess.TimeoutExpired:
                # Handle timeout case
                with open(output_file, "a") as f:
                    f.write(f"## Test: {test_file} - Method: {method}\n")
                    f.write(f"TIMEOUT: Execution exceeded {timeout} seconds\n")
                    f.write("\n" + "-" * 80 + "\n\n")

    print(f"\nCompleted all tests. Results saved to {output_file}")
    return 0


def extract_test_info(content):
    """Extract test results from content with improved solution detection"""
    test_pattern = r"## Test: (.*?) - Method: (.*?)\n(.*?)\n(.*?)\n(.*?)\n"
    test_results = {}

    matches = re.findall(test_pattern, content, re.DOTALL)
    for match in matches:
        test_file, method, line1, line2, line3 = match
        key = (test_file, method)

        # Check if this was a timeout
        if line1.strip().startswith("TIMEOUT:"):
            test_results[key] = {
                "status": "timeout",
                "goal": None,
                "nodes_expanded": None,
                "path": None,
            }
            continue

        # Check if a solution was found by looking for the "goal nodes_expanded" format in line2
        has_solution = bool(re.match(r"^\d+ \d+$", line2.strip()))

        if has_solution:
            # Extract goal and nodes expanded
            goal, nodes_expanded = line2.strip().split()

            # Extract path
            try:
                path_str = line3.strip()
                if path_str.startswith("[") and path_str.endswith("]"):
                    # Convert the string representation of a list to a real list
                    path = [int(x.strip()) for x in path_str[1:-1].split(",")]
                else:
                    # If not in list format, split by spaces
                    path = [int(x) for x in path_str.split()]
            except Exception:
                path = None

            test_results[key] = {
                "status": "solution_found",
                "goal": goal,
                "nodes_expanded": nodes_expanded,
                "path": path,
            }
        else:
            test_results[key] = {
                "status": "no_solution",
                "goal": None,
                "nodes_expanded": None,
                "path": None,
            }

    return test_results


def compare_solutions(solution_file, student_file, output_file):
    """Compare solution with student output, excluding CUS1 and CUS2 methods"""
    # Read the files
    with open(solution_file, "r") as f:
        solution_content = f.read()

    with open(student_file, "r") as f:
        student_content = f.read()

    # Extract test results
    solution_results = extract_test_info(solution_content)
    student_results = extract_test_info(student_content)

    # Only compare BFS, DFS, AS, and GBFS
    methods_to_compare = ["BFS", "DFS", "AS", "GBFS"]
    differences = []

    for key in solution_results:
        test_file, method = key
        # Skip CUS1 and CUS2
        if method not in methods_to_compare:
            continue

        solution = solution_results[key]

        # Check if student has this test result
        if key not in student_results:
            differences.append(
                f"MISSING: Test {test_file} - Method {method} is missing from student output\n\n----"
            )
            continue

        student = student_results[key]

        # Check for timeout in student implementation
        if student["status"] == "timeout":
            diff = [
                f"TIMEOUT ERROR: Test {test_file} - Method {method}",
                f"  Student implementation exceeded timeout limit\n\n----",
            ]
            differences.append("\n".join(diff))
            continue

        # Compare solution status
        if solution["status"] != student["status"]:
            diff = [
                f"STATUS MISMATCH: Test {test_file} - Method {method}",
                f"  Expected: {solution['status']}",
                f"  Got: {student['status']}\n\n----",
            ]
            differences.append("\n".join(diff))
            continue

        # If both found no solution, that's fine
        if solution["status"] == "no_solution" and student["status"] == "no_solution":
            continue

        # Compare goal node
        if solution["goal"] != student["goal"]:
            diff = [
                f"GOAL MISMATCH: Test {test_file} - Method {method}",
                f"  Expected: {solution['goal']}",
                f"  Got: {student['goal']}",
            ]
            differences.append("\n".join(diff))

        # Compare path
        if solution["path"] != student["path"]:
            diff = [
                f"PATH MISMATCH: Test {test_file} - Method {method}",
                f"  Expected: {solution['path']}",
                f"  Got: {student['path']}",
            ]
            differences.append("\n".join(diff))

        # Add separator at the end of each test case with differences
        if solution["goal"] != student["goal"] or solution["path"] != student["path"]:
            differences.append("\n----")

    # Write results to output file
    with open(output_file, "w") as f:
        if differences:
            f.write("DIFFERENCES FOUND:\n\n")
            f.write("\n\n".join(differences))
            f.write(
                "\n\nNote: CUS1 and CUS2 algorithms were not automatically compared.\n"
            )
        else:
            f.write("All BFS, DFS, AS, and GBFS tests passed!\n")
            f.write(
                "\nNote: CUS1 and CUS2 algorithms were not automatically compared.\n"
            )

    return len(differences) == 0  # Return True if no differences


def main():
    # Run tests and generate student output
    student_output_file = "3_student_output.txt"
    print(f"Generating student output to {student_output_file}...")
    run_search_on_all_testcases(student_output_file)

    # Compare with solution
    solution_file = "2_solution.txt"
    output_file = "4_test_result.txt"

    print(f"Comparing student output with solution...")
    if os.path.exists(solution_file):
        passed = compare_solutions(solution_file, student_output_file, output_file)
        if passed:
            print(
                f"All BFS, DFS, AS, and GBFS tests passed! See {output_file} for details."
            )
        else:
            print(
                f"Differences found in BFS, DFS, AS, or GBFS tests. See {output_file} for details."
            )
    else:
        print(f"Error: Solution file '{solution_file}' not found")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

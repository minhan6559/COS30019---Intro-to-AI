from src.problem.multigoal_graph_problem import MultigoalGraphProblem
from src.search_algorithm.search_algorithm import (
    BreadthFirstSearch,
    DepthFirstSearch,
    AStarSearch,
    GreedyBestFirstSearch,
    UniformCostSearch,
    BULBSearch,
    UniformCostSearch,
    BULBSearch,
)


def gay():
    filename = "PathFinder-test.txt"

    # Create a random graph problem
    original_problem = MultigoalGraphProblem.from_file(filename)

    search_algo = UniformCostSearch()

    result = search_algo.search(original_problem)

    print("Result:", result)


if __name__ == "__main__":
    gay()

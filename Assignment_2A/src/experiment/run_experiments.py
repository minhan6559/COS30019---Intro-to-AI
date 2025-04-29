#!/usr/bin/env python3
import os
import sys
import argparse
import time
import json
import glob
import logging
import traceback
import tracemalloc
from pathlib import Path
from datetime import datetime

# Add the project root to the path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.problem.multigoal_graph_problem import MultigoalGraphProblem
from src.search_algorithm.search_algorithm import (
    BreadthFirstSearch,
    DepthFirstSearch,
    AStarSearch,
    GreedyBestFirstSearch,
    UniformCostSearch,
    BULBSearch,
)


def setup_logging(timestamp, log_dir="logs"):
    """Set up logging configuration"""
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create checkpoint-specific log directory
    checkpoint_log_dir = os.path.join(log_dir, timestamp)
    if not os.path.exists(checkpoint_log_dir):
        os.makedirs(checkpoint_log_dir)

    # Configure logging
    log_file = os.path.join(checkpoint_log_dir, f"run_experiments.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("run_experiments")
    logger.info(f"Logs will be saved to: {log_file}")
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run search algorithm experiments")
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        help="Only run experiments on graphs of this size",
    )
    parser.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[1000, 3000, 5000, 8000, 10000],
        help="List of node counts for graph selection",
    )
    parser.add_argument(
        "--graphs-per-size",
        type=int,
        default=3,
        help="Number of graphs per node count",
    )
    parser.add_argument(
        "--graphs-dir",
        default="data/graphs",
        help="Directory containing the generated graphs",
    )
    parser.add_argument(
        "--results-dir",
        default="data/results",
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Timestamp to use for this checkpoint (default: current time)",
    )
    parser.add_argument(
        "--graphs-checkpoint",
        help="Specific graphs checkpoint to use (timestamp)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory to save log files (default: logs)",
    )
    return parser.parse_args()


def ensure_dir(directory):
    """Ensure that a directory exists"""
    os.makedirs(directory, exist_ok=True)


def get_latest_checkpoint(base_dir, logger=None):
    """Get the latest checkpoint from the base directory"""
    # Find all subdirectories in the base directory
    checkpoints = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not checkpoints:
        if logger:
            logger.warning(f"No checkpoint directories found in {base_dir}")
        return None

    # Sort checkpoints by name (which should be timestamps)
    latest_checkpoint = sorted(checkpoints)[-1]

    if logger:
        logger.info(f"Found most recent checkpoint directory: {latest_checkpoint}")

    return latest_checkpoint


def get_algorithms():
    """Return a dictionary of search algorithms to test"""
    return {
        "BFS": BreadthFirstSearch(),
        "DFS": DepthFirstSearch(),
        "A*": AStarSearch(),
        "Greedy": GreedyBestFirstSearch(),
        "UCS": UniformCostSearch(),
        "BULB": BULBSearch(),
    }


def run_experiment(problem, algorithm_name, algorithm, logger=None, track_memory=False):
    """
    Run a search algorithm on a problem

    Args:
        problem: The problem to solve
        algorithm_name: Name of the algorithm
        algorithm: The search algorithm instance
        logger: Logger instance
        track_memory: Whether to track memory usage (adds overhead to runtime)

    Returns:
        dict: Results dictionary with metrics
    """
    if logger:
        if track_memory:
            logger.info(f"  Running {algorithm_name} WITH memory tracking...")
        else:
            logger.info(f"  Running {algorithm_name} for runtime measurement only...")

    start_time = time.time() * 1000  # Convert to milliseconds

    # Only start memory tracking if requested
    if track_memory:
        tracemalloc.start()
        peak_memory_kb = 0

    try:
        result_tuple = algorithm.search(problem)

        # Capture peak memory usage if tracking
        if track_memory:
            current, peak = tracemalloc.get_traced_memory()
            peak_memory_kb = peak / 1024  # Convert to KB
            # Stop memory tracking
            tracemalloc.stop()

        end_time = time.time() * 1000  # Convert to milliseconds
        runtime_ms = end_time - start_time

        if result_tuple and result_tuple[0]:  # If solution found
            result_node, expanded_count, created_count = result_tuple

            path_nodes = result_node.path_states()
            path_cost = result_node.path_cost

            if logger:
                if track_memory:
                    logger.info(
                        f"    {algorithm_name}: Found path of length {len(path_nodes)} "
                        f"with cost {path_cost:.2f} in {runtime_ms:.2f}ms"
                    )
                    logger.debug(
                        f"    Nodes expanded: {expanded_count}, created: {created_count}"
                    )
                    logger.debug(f"    Peak memory usage: {peak_memory_kb:.2f} KB")
                else:
                    logger.info(
                        f"    {algorithm_name}: Runtime measurement: {runtime_ms:.2f}ms"
                    )

            if track_memory:
                return {
                    "success": True,
                    "path_cost": path_cost,
                    "path_length": len(path_nodes),
                    "nodes_expanded": expanded_count,
                    "nodes_created": created_count,
                    "peak_memory_kb": peak_memory_kb,
                }
            else:
                return {
                    "success": True,
                    "runtime_ms": runtime_ms,
                }
        else:
            # No solution found, but algorithm completed
            expanded_count = result_tuple[1] if result_tuple else 0
            created_count = result_tuple[2] if result_tuple else 0

            if logger:
                if track_memory:
                    logger.info(
                        f"    {algorithm_name}: No solution found after {runtime_ms:.2f}ms"
                    )
                    logger.debug(
                        f"    Nodes expanded: {expanded_count}, created: {created_count}"
                    )
                    logger.debug(f"    Peak memory usage: {peak_memory_kb:.2f} KB")
                else:
                    logger.info(
                        f"    {algorithm_name}: No solution (runtime: {runtime_ms:.2f}ms)"
                    )

            if track_memory:
                return {
                    "success": False,
                    "path_cost": float("inf"),
                    "path_length": 0,
                    "nodes_expanded": expanded_count,
                    "nodes_created": created_count,
                    "peak_memory_kb": peak_memory_kb,
                }
            else:
                return {
                    "success": False,
                    "runtime_ms": runtime_ms,
                }

    except Exception as e:
        # Make sure to stop tracemalloc even on exception
        if track_memory:
            current, peak = tracemalloc.get_traced_memory()
            peak_memory_kb = peak / 1024  # Convert to KB
            tracemalloc.stop()

        end_time = time.time() * 1000  # Convert to milliseconds
        runtime_ms = end_time - start_time

        if logger:
            logger.error(f"    Error running {algorithm_name}: {str(e)}")
            logger.debug(traceback.format_exc())
            if track_memory:
                logger.debug(f"    Peak memory usage: {peak_memory_kb:.2f} KB")

        if track_memory:
            return {
                "success": False,
                "path_cost": float("inf"),
                "path_length": 0,
                "nodes_expanded": 0,
                "nodes_created": 0,
                "peak_memory_kb": peak_memory_kb,
                "error": str(e),
            }
        else:
            return {
                "success": False,
                "runtime_ms": runtime_ms,
                "error": str(e),
            }


def main():
    args = parse_args()

    # Set up logging
    logger = setup_logging(args.timestamp, args.log_dir)
    logger.info(
        f"Experiment run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Command line arguments: {sys.argv}")

    # Ensure the base results directory exists
    base_results_dir = os.path.abspath(args.results_dir)
    ensure_dir(base_results_dir)
    logger.info(f"Results base directory: {base_results_dir}")

    # Create a timestamp-based checkpoint directory for results
    results_checkpoint_dir = os.path.join(base_results_dir, args.timestamp)
    ensure_dir(results_checkpoint_dir)
    logger.info(f"Created results checkpoint directory: {results_checkpoint_dir}")

    # Get the base graphs directory
    base_graphs_dir = os.path.abspath(args.graphs_dir)
    logger.info(f"Graphs base directory: {base_graphs_dir}")

    # Determine which graphs checkpoint to use
    graphs_checkpoint = args.graphs_checkpoint
    if not graphs_checkpoint:
        graphs_checkpoint = get_latest_checkpoint(base_graphs_dir, logger)
        if not graphs_checkpoint:
            logger.error(f"No graph checkpoints found in {base_graphs_dir}")
            return 1

    graphs_dir = os.path.join(base_graphs_dir, graphs_checkpoint)
    if not os.path.exists(graphs_dir):
        logger.error(f"Graph checkpoint directory not found: {graphs_dir}")
        return 1

    logger.info(f"Using graph checkpoint: {graphs_checkpoint}")
    logger.info(f"Results will be saved to checkpoint: {args.timestamp}")

    # Select which node counts to process
    node_counts = args.node_counts
    if args.size:
        # Map size categories to node counts
        size_map = {
            "small": [1000],
            "medium": [3000, 5000],
            "large": [8000, 10000],
        }
        node_counts = [n for n in node_counts if n in size_map[args.size]]
        logger.info(f"Filtered node counts by size '{args.size}': {node_counts}")
    else:
        logger.info(f"Using all node counts: {node_counts}")

    # Initialize results structure
    all_results = {
        "metadata": {
            "timestamp": args.timestamp,
            "graphs_checkpoint": graphs_checkpoint,
            "node_counts": node_counts,
            "graphs_per_size": args.graphs_per_size,
        },
        "results": {},
    }

    # Get algorithms to test
    algorithms = get_algorithms()
    logger.info(f"Using search algorithms: {list(algorithms.keys())}")

    # Process each node count
    for node_count in node_counts:
        logger.info(f"\nProcessing graphs with {node_count} nodes:")
        node_dir = os.path.join(graphs_dir, f"nodes_{node_count}")

        if not os.path.exists(node_dir):
            logger.warning(f"  Directory not found: {node_dir}")
            continue

        # Initialize results for this node count
        all_results["results"][str(node_count)] = {}

        # Find all graph files for this node count
        graph_files = glob.glob(os.path.join(node_dir, f"graph_{node_count}_*.txt"))
        graph_files.sort()
        logger.info(f"  Found {len(graph_files)} graph files in {node_dir}")

        # Process each graph file
        for graph_file in graph_files:
            graph_basename = os.path.basename(graph_file)
            graph_id = os.path.splitext(graph_basename)[0]

            logger.info(f"  Processing graph: {graph_basename}")

            # Load the graph problem
            try:
                problem = MultigoalGraphProblem.from_file(graph_file)
                logger.info(f"    Initial: {problem.initial}, Goals: {problem.goals}")
                logger.debug(f"    Graph has {len(problem.graph.nodes())} nodes")
            except Exception as e:
                logger.error(f"    Error loading graph: {str(e)}")
                logger.debug(traceback.format_exc())
                continue

            # Initialize results for this graph
            all_results["results"][str(node_count)][graph_id] = {}

            # Run each algorithm twice - once for runtime, once for memory usage
            for alg_name, algorithm in algorithms.items():
                logger.info(f"  Testing algorithm: {alg_name}")

                # First run - runtime only (no memory tracking)
                runtime_results = run_experiment(
                    problem,
                    alg_name,
                    algorithm,
                    logger,
                    track_memory=False,
                )

                # Second run - with memory tracking for all other metrics
                metrics_results = run_experiment(
                    problem,
                    alg_name,
                    algorithm,
                    logger,
                    track_memory=True,
                )

                # Merge results, using runtime from the first run
                combined_results = metrics_results.copy()
                if "runtime_ms" in runtime_results:
                    combined_results["runtime_ms"] = runtime_results["runtime_ms"]

                all_results["results"][str(node_count)][graph_id][
                    alg_name
                ] = combined_results

    # Save results to file
    results_file = os.path.join(results_checkpoint_dir, f"search_results.json")

    logger.info(f"\nSaving results to {results_file}")
    try:
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        logger.debug(traceback.format_exc())

    logger.info("\nExperiments completed successfully!")
    logger.info(f"Results checkpoint created: {args.timestamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

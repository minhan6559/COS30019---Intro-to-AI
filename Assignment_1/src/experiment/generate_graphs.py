#!/usr/bin/env python3
import os
import sys
import argparse
import time
import random
import glob
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to the path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.problem.multigoal_graph_problem import MultigoalGraphProblem


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
    log_file = os.path.join(checkpoint_log_dir, f"generate_graphs.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("generate_graphs")
    logger.info(f"Logs will be saved to: {log_file}")
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate test graphs for search experiments"
    )
    parser.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[1000, 3000, 5000, 8000, 10000],
        help="List of node counts for graph generation",
    )
    parser.add_argument(
        "--graphs-per-size",
        type=int,
        default=3,
        help="Number of graphs to generate per node count",
    )
    parser.add_argument(
        "--min-edges",
        type=int,
        default=10,
        help="Minimum edges per node",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=20,
        help="Maximum edges per node",
    )
    parser.add_argument(
        "--output-dir",
        default="data/graphs",
        help="Directory to save generated graphs",
    )
    parser.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Timestamp to use for this checkpoint (default: current time)",
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


def generate_graph(num_nodes, min_edges, max_edges, seed=None, logger=None):
    """Generate a random graph with specified parameters"""
    if seed is not None:
        random.seed(seed)

    if logger:
        logger.info(
            f"Generating graph with {num_nodes} nodes, {min_edges}-{max_edges} edges per node..."
        )
    start_time = time.time()

    # Grid size scales with node count for better spacing
    grid_size = num_nodes * 3

    try:
        problem = MultigoalGraphProblem.random(
            num_nodes=num_nodes,
            min_edges_per_node=min_edges,
            max_edges_per_node=max_edges,
            grid_size=grid_size,
            num_destinations=1,  # Multiple destinations for more interesting problems
            ensure_connectivity=True,
        )

        # Calculate total edges and average edges per node
        total_edges = sum(
            len(neighbors) for neighbors in problem.graph.graph_dict.values()
        )
        avg_edges_per_node = total_edges / len(problem.graph.nodes())

        elapsed_time = time.time() - start_time
        if logger:
            logger.info(f"  Generation completed in {elapsed_time:.2f} seconds")
            logger.info(
                f"  Graph has {len(problem.graph.nodes())} nodes and {total_edges} edges"
            )
            logger.info(f"  Average edges per node: {avg_edges_per_node:.2f}")

        return problem
    except Exception as e:
        if logger:
            logger.error(f"Graph generation failed: {str(e)}")
        raise


def main():
    args = parse_args()

    # Set up logging
    logger = setup_logging(args.timestamp, args.log_dir)
    logger.info(
        f"Graph generation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Command line arguments: {sys.argv}")

    # Ensure the base directory exists
    base_dir = os.path.abspath(args.output_dir)
    ensure_dir(base_dir)

    # Create a timestamp-based checkpoint directory
    checkpoint_dir = os.path.join(base_dir, args.timestamp)
    ensure_dir(checkpoint_dir)
    logger.info(f"Created checkpoint directory: {checkpoint_dir}")

    logger.info(f"Generating graphs with parameters:")
    logger.info(f"  Node counts: {args.node_counts}")
    logger.info(f"  Graphs per size: {args.graphs_per_size}")
    logger.info(f"  Edge range: {args.min_edges}-{args.max_edges} per node")
    logger.info(f"  Output directory: {checkpoint_dir}")
    logger.info(f"  Checkpoint timestamp: {args.timestamp}")

    # Track overall statistics
    all_graphs_avg_edges = []

    # Generate graphs for each node count
    for node_count in args.node_counts:
        node_dir = os.path.join(checkpoint_dir, f"nodes_{node_count}")
        ensure_dir(node_dir)
        logger.info(f"Processing node count: {node_count}")

        node_count_avg_edges = []

        for i in range(1, args.graphs_per_size + 1):
            # Use a deterministic seed if want reproducibility
            # seed = node_count * 100 + i
            seed = None

            logger.info(
                f"  Generating graph {i}/{args.graphs_per_size} with seed {seed}"
            )

            try:
                # Generate the graph
                problem = generate_graph(
                    num_nodes=node_count,
                    min_edges=args.min_edges,
                    max_edges=args.max_edges,
                    seed=seed,
                    logger=logger,
                )

                # Calculate average edges per node for statistics
                total_edges = sum(
                    len(neighbors) for neighbors in problem.graph.graph_dict.values()
                )
                avg_edges = total_edges / len(problem.graph.nodes())
                node_count_avg_edges.append(avg_edges)
                all_graphs_avg_edges.append(avg_edges)

                # Save the graph to a file
                filename = f"graph_{node_count}_{i}.txt"
                filepath = os.path.join(node_dir, filename)

                logger.info(f"  Saving graph to {filepath}")
                MultigoalGraphProblem.to_file(problem, filepath)
                logger.info(f"  Graph saved successfully")

            except Exception as e:
                logger.error(f"  Failed to generate or save graph: {str(e)}")
                continue

        # Log average edges for this node count
        if node_count_avg_edges:
            avg = sum(node_count_avg_edges) / len(node_count_avg_edges)
            logger.info(
                f"  Average edges per node for {node_count}-node graphs: {avg:.2f}"
            )

    # Log overall average
    if all_graphs_avg_edges:
        overall_avg = sum(all_graphs_avg_edges) / len(all_graphs_avg_edges)
        logger.info(
            f"\nOverall average edges per node across all graphs: {overall_avg:.2f}"
        )

    logger.info("\nAll graphs generated successfully!")
    logger.info(f"Checkpoint created: {args.timestamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

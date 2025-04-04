#!/usr/bin/env python3
import os
import sys
import argparse
import json
import glob
import logging
import traceback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


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
    log_file = os.path.join(checkpoint_log_dir, f"analyze_results.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("analyze_results")
    logger.info(f"Logs will be saved to: {log_file}")
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze search algorithm experiment results"
    )
    parser.add_argument("--results", help="Specific results file to analyze")
    parser.add_argument(
        "--results-dir",
        default="data/results",
        help="Directory containing result files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/analysis",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--log-scale", action="store_true", help="Use logarithmic scale for charts"
    )
    parser.add_argument(
        "--no-log-scale",
        dest="log_scale",
        action="store_false",
        help="Use linear scale for charts",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["path_cost", "path_length", "nodes_expanded", "runtime_ms"],
        help="Metrics to include in analysis",
    )
    parser.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Timestamp to use for this checkpoint (default: current time)",
    )
    parser.add_argument(
        "--results-checkpoint",
        help="Specific results checkpoint to use (timestamp)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory to save log files (default: logs)",
    )
    parser.set_defaults(log_scale=True)
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


def load_results(results_file, logger=None):
    """Load results from JSON file"""
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
        if logger:
            logger.info(f"Successfully loaded results from {results_file}")
            if "metadata" in data:
                logger.info(f"Results metadata: {data['metadata']}")
        return data
    except Exception as e:
        if logger:
            logger.error(f"Failed to load results file: {str(e)}")
            logger.debug(traceback.format_exc())
        raise


def calculate_means(results_data, logger=None):
    """
    Calculate means of metrics across graphs of the same size

    Args:
        results_data (dict): The results data structure

    Returns:
        dict: Dictionary of means by node count and algorithm
    """
    if logger:
        logger.info("Calculating means across graphs of the same size")

    means = {}

    for node_count, graphs in results_data["results"].items():
        if logger:
            logger.info(f"Processing node count: {node_count}")

        means[node_count] = {}

        # Get all algorithms
        algorithms = set()
        for graph_data in graphs.values():
            algorithms.update(graph_data.keys())

        if logger:
            logger.debug(f"  Found algorithms: {algorithms}")

        # Initialize mean dictionaries for each algorithm
        for alg in algorithms:
            means[node_count][alg] = {
                "path_cost": [],
                "path_length": [],
                "nodes_expanded": [],
                "nodes_created": [],
                "runtime_ms": [],
                "success_rate": 0,
            }

        # Calculate means across graphs
        for graph_id, graph_data in graphs.items():
            for alg, results in graph_data.items():
                if results["success"]:
                    means[node_count][alg]["path_cost"].append(results["path_cost"])
                    means[node_count][alg]["path_length"].append(results["path_length"])
                    means[node_count][alg]["nodes_expanded"].append(
                        results["nodes_expanded"]
                    )
                    means[node_count][alg]["nodes_created"].append(
                        results.get("nodes_created", 0)
                    )
                    means[node_count][alg]["runtime_ms"].append(results["runtime_ms"])
                    means[node_count][alg]["success_rate"] += 1

        # Calculate final means and success rates
        for alg in algorithms:
            total_graphs = len(graphs)
            success_count = means[node_count][alg]["success_rate"]
            means[node_count][alg]["success_rate"] /= total_graphs

            if logger:
                logger.debug(
                    f"  {alg} success rate: {success_count}/{total_graphs} = {means[node_count][alg]['success_rate']:.2f}"
                )

            # Calculate means only if there are successful runs
            for metric in [
                "path_cost",
                "path_length",
                "nodes_expanded",
                "nodes_created",
                "runtime_ms",
            ]:
                values = means[node_count][alg][metric]
                if values:
                    means[node_count][alg][metric] = np.mean(values)
                    if logger:
                        logger.debug(
                            f"  {alg} mean {metric}: {means[node_count][alg][metric]:.2f}"
                        )
                else:
                    means[node_count][alg][metric] = np.nan
                    if logger:
                        logger.debug(f"  {alg} mean {metric}: N/A (no successful runs)")

    return means


def generate_charts(means, metrics, output_dir, use_log_scale=True, logger=None):
    """
    Generate charts for the specified metrics

    Args:
        means (dict): The calculated means
        metrics (list): List of metrics to chart
        output_dir (str): Directory to save charts
        use_log_scale (bool): Whether to use logarithmic scale
    """
    if logger:
        logger.info(f"Generating charts for metrics: {metrics}")
        logger.info(f"Using log scale: {use_log_scale}")

    # Get node counts and algorithms
    node_counts = sorted([int(nc) for nc in means.keys()])
    if logger:
        logger.debug(f"Node counts: {node_counts}")

    # Get a list of all algorithms across all node counts
    all_algorithms = set()
    for nc_data in means.values():
        all_algorithms.update(nc_data.keys())
    all_algorithms = sorted(all_algorithms)
    if logger:
        logger.debug(f"Algorithms: {all_algorithms}")

    # Define colors and markers for algorithms
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_algorithms)))
    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "<",
        ">",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "d",
        "|",
    ]

    # Create charts for each metric
    for metric in metrics:
        if logger:
            logger.info(f"Generating chart for metric: {metric}")

        plt.figure(figsize=(12, 8))

        # Plot data for each algorithm
        for i, alg in enumerate(all_algorithms):
            # Extract data for this algorithm
            x_values = []
            y_values = []

            for nc in node_counts:
                if str(nc) in means and alg in means[str(nc)]:
                    value = means[str(nc)][alg][metric]
                    if not np.isnan(value):
                        x_values.append(nc)
                        y_values.append(value)

            if x_values:
                if logger:
                    logger.debug(f"  {alg} data points: {len(x_values)}")

                plt.plot(
                    x_values,
                    y_values,
                    label=alg,
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    linewidth=2,
                    markersize=8,
                )
            else:
                if logger:
                    logger.debug(f"  No valid data points for {alg}")

        # Set chart properties
        plt.title(f'Mean {metric.replace("_", " ").title()} by Node Count', fontsize=16)
        plt.xlabel("Number of Nodes", fontsize=14)

        # Add "(log scale)" to Y axis title if using log scale
        y_label = metric.replace("_", " ").title()
        if use_log_scale:
            y_label += " (log scale)"
        plt.ylabel(y_label, fontsize=14)

        # Use log scale if specified
        if use_log_scale:
            plt.yscale("log")
            plt.xscale("log")

        # Set x-axis ticks to show exact node counts
        plt.xticks(node_counts, [str(nc) for nc in node_counts])

        # Add grid
        plt.grid(True, which="both", ls="--", alpha=0.3)

        # Add legend
        plt.legend(fontsize=12)

        # Adjust layout
        plt.tight_layout()

        # Save chart
        chart_file = os.path.join(output_dir, f"chart_{metric}.png")
        plt.savefig(chart_file, dpi=300)
        plt.close()

        if logger:
            logger.info(f"Saved chart: {chart_file}")

    # Generate success rate chart
    if logger:
        logger.info("Generating success rate chart")

    plt.figure(figsize=(12, 8))

    for i, alg in enumerate(all_algorithms):
        x_values = []
        y_values = []

        for nc in node_counts:
            if str(nc) in means and alg in means[str(nc)]:
                x_values.append(nc)
                y_values.append(
                    means[str(nc)][alg]["success_rate"] * 100
                )  # Convert to percentage

        if x_values:
            plt.plot(
                x_values,
                y_values,
                label=alg,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linewidth=2,
                markersize=8,
            )

    plt.title("Success Rate by Node Count", fontsize=16)
    plt.xlabel("Number of Nodes", fontsize=14)
    plt.ylabel("Success Rate (%)", fontsize=14)

    if use_log_scale:
        plt.xscale("log")  # Only x-axis log for success rate
        # Set x-axis ticks to show exact node counts
        plt.xticks(node_counts, [str(nc) for nc in node_counts])

    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    chart_file = os.path.join(output_dir, "chart_success_rate.png")
    plt.savefig(chart_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"Saved chart: {chart_file}")


def main():
    args = parse_args()

    # Set up logging
    logger = setup_logging(args.timestamp, args.log_dir)
    logger.info(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Command line arguments: {sys.argv}")

    # Ensure the base analysis directory exists
    base_output_dir = os.path.abspath(args.output_dir)
    ensure_dir(base_output_dir)
    logger.info(f"Analysis base directory: {base_output_dir}")

    # Create a timestamp-based checkpoint directory for analysis results
    output_checkpoint_dir = os.path.join(base_output_dir, args.timestamp)
    ensure_dir(output_checkpoint_dir)
    logger.info(f"Created analysis checkpoint directory: {output_checkpoint_dir}")

    # Create a "latest" file to track the most recent analysis checkpoint
    latest_output_file = os.path.join(base_output_dir, "latest")
    with open(latest_output_file, "w") as f:
        f.write(args.timestamp)
    logger.info(f"Updated latest analysis checkpoint pointer to: {args.timestamp}")

    # Get the base results directory
    base_results_dir = os.path.abspath(args.results_dir)
    logger.info(f"Results base directory: {base_results_dir}")

    # Determine which results checkpoint to use
    if args.results:
        # If a specific results file is provided, use it directly
        results_files = [args.results]
        logger.info(f"Using specific results file: {args.results}")
    else:
        # Otherwise look for a checkpoint
        results_checkpoint = args.results_checkpoint
        if not results_checkpoint:
            results_checkpoint = get_latest_checkpoint(base_results_dir, logger)
            if not results_checkpoint:
                logger.error(f"No results checkpoints found in {base_results_dir}")
                return 1

        results_dir = os.path.join(base_results_dir, results_checkpoint)
        if not os.path.exists(results_dir):
            logger.error(f"Results checkpoint directory not found: {results_dir}")
            return 1

        # Look for search_results.json in the checkpoint directory
        results_files = [os.path.join(results_dir, "search_results.json")]
        if not os.path.exists(results_files[0]):
            # Try to find any JSON files in the directory
            results_files = glob.glob(os.path.join(results_dir, "*.json"))

        if not results_files:
            logger.error(f"No results files found in checkpoint: {results_dir}")
            return 1

        logger.info(f"Using results checkpoint: {results_checkpoint}")
        logger.info(f"Found results files: {results_files}")

    logger.info(f"Analysis results will be saved to checkpoint: {args.timestamp}")

    # Process each results file
    for results_file in results_files:
        logger.info(f"Analyzing results file: {results_file}")

        # Load results
        try:
            results_data = load_results(results_file, logger)
        except Exception as e:
            logger.error(f"Error loading results file: {str(e)}")
            continue

        # Calculate means
        means = calculate_means(results_data, logger)

        # Generate charts
        generate_charts(
            means, args.metrics, output_checkpoint_dir, args.log_scale, logger
        )

        # Save means to file
        means_file = os.path.join(output_checkpoint_dir, "means.json")
        try:
            with open(means_file, "w") as f:
                json.dump(means, f, indent=2)
            logger.info(f"Saved means to: {means_file}")
        except Exception as e:
            logger.error(f"Failed to save means file: {str(e)}")
            logger.debug(traceback.format_exc())

        # Save source data info
        source_info = {
            "timestamp": args.timestamp,
            "source_file": results_file,
            "source_data": results_data.get("metadata", {}),
        }
        source_file = os.path.join(output_checkpoint_dir, "source_info.json")
        try:
            with open(source_file, "w") as f:
                json.dump(source_info, f, indent=2)
            logger.info(f"Saved source info to: {source_file}")
        except Exception as e:
            logger.error(f"Failed to save source info file: {str(e)}")
            logger.debug(traceback.format_exc())

    logger.info("\nAnalysis completed successfully!")
    logger.info(f"Analysis checkpoint created: {args.timestamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

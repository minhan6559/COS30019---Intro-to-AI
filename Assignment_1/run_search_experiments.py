#!/usr/bin/env python3
import os
import sys
import argparse
import time
import subprocess
import logging
from datetime import datetime


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
    log_file = os.path.join(checkpoint_log_dir, f"experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("experiment")
    logger.info(f"Logs will be saved to: {log_file}")
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run search algorithm experiments")

    # Main action arguments
    action_group = parser.add_argument_group("Actions")
    action_group.add_argument(
        "--generate", action="store_true", help="Generate test graphs"
    )
    action_group.add_argument("--run", action="store_true", help="Run experiments")
    action_group.add_argument("--analyze", action="store_true", help="Analyze results")
    action_group.add_argument(
        "--all", action="store_true", help="Run the complete experiment pipeline"
    )

    # Graph generation options
    generate_group = parser.add_argument_group("Graph Generation Options (--generate)")
    generate_group.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[1000, 3000, 5000, 8000, 10000],
        help="List of node counts for graph generation (default: 1000 3000 5000 8000 10000)",
    )
    generate_group.add_argument(
        "--graphs-per-size",
        type=int,
        default=3,
        help="Number of graphs to generate per node count (default: 3)",
    )
    generate_group.add_argument(
        "--min-edges",
        type=int,
        default=10,
        help="Minimum edges per node (default: 10)",
    )
    generate_group.add_argument(
        "--max-edges",
        type=int,
        default=20,
        help="Maximum edges per node (default: 20)",
    )

    # Run experiment options
    run_group = parser.add_argument_group("Experiment Run Options (--run)")
    run_group.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        help="Only run experiments on graphs of this size: small=1000, medium=3000,5000, large=8000,10000 nodes",
    )
    run_group.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each algorithm run (default: 300)",
    )
    run_group.add_argument(
        "--run-node-counts",
        type=int,
        nargs="+",
        help="Specific node counts to use for experiments (defaults to same as --node-counts if not specified)",
    )

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options (--analyze)")
    analysis_group.add_argument(
        "--log-scale",
        action="store_true",
        default=True,
        help="Use logarithmic scale for visualizations (default: True)",
    )
    analysis_group.add_argument(
        "--results",
        help="Specific results file to analyze instead of using checkpoints",
    )

    # Checkpoint options
    checkpoint_group = parser.add_argument_group("Checkpoint Options")
    checkpoint_group.add_argument(
        "--checkpoint",
        help="Use specific checkpoint (timestamp) for all steps",
    )
    checkpoint_group.add_argument(
        "--graphs-checkpoint",
        help="Use specific checkpoint for graph data (--run step)",
    )
    checkpoint_group.add_argument(
        "--results-checkpoint",
        help="Use specific checkpoint for experiment results (--analyze step)",
    )

    # General options
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument(
        "--log-dir",
        default="logs",
        help="Directory to save log files (default: logs)",
    )

    return parser.parse_args()


def run_command(command, logger):
    """Run a command and return its exit code"""
    try:
        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        logger.info(
            f"Command completed successfully with return code {result.returncode}"
        )
        # Log stdout and stderr at different levels
        if result.stdout:
            logger.debug(f"Command stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Command stderr: {result.stderr}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            logger.debug(f"Command stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Command stderr: {e.stderr}")
        return e.returncode


def main():
    args = parse_args()

    # If no options specified, show help
    if not (args.generate or args.run or args.analyze or args.all):
        print("No actions specified. Use --help to see available options.")
        return 1

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up logging
    logger = setup_logging(timestamp, args.log_dir)
    logger.info(
        f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Command line arguments: {sys.argv}")

    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Working directory: {current_dir}")

    # Define script paths
    generate_script = os.path.join(
        current_dir, "src", "experiment", "generate_graphs.py"
    )
    run_script = os.path.join(current_dir, "src", "experiment", "run_experiments.py")
    analyze_script = os.path.join(
        current_dir, "src", "experiment", "analyze_results.py"
    )

    # Global checkpoint for all steps (if provided)
    global_checkpoint = args.checkpoint
    if global_checkpoint:
        logger.info(f"Using global checkpoint: {global_checkpoint}")

    # Run the pipeline
    if args.generate or args.all:
        logger.info("=" * 50)
        logger.info("STEP 1: Generating test graphs")
        logger.info("=" * 50)

        # Build command with new parameters
        generate_command_args = [sys.executable, generate_script]
        generate_command_args.extend(
            ["--node-counts"] + [str(n) for n in args.node_counts]
        )
        generate_command_args.extend(["--graphs-per-size", str(args.graphs_per_size)])
        generate_command_args.extend(["--min-edges", str(args.min_edges)])
        generate_command_args.extend(["--max-edges", str(args.max_edges)])
        generate_command_args.extend(["--log-dir", args.log_dir])

        # Add checkpoint info
        generate_command_args.extend(["--timestamp", timestamp])

        exit_code = run_command(generate_command_args, logger)
        if exit_code != 0:
            logger.error("Graph generation failed. Exiting.")
            return exit_code

    if args.run or args.all:
        logger.info("=" * 50)
        logger.info("STEP 2: Running experiments")
        logger.info("=" * 50)

        # Build command with optional arguments
        run_command_args = [sys.executable, run_script]
        if args.size:
            run_command_args.extend(["--size", args.size])
        if args.timeout:
            run_command_args.extend(["--timeout", str(args.timeout)])

        # Use run-node-counts if specified, otherwise fall back to node-counts
        node_counts_for_run = (
            args.run_node_counts
            if args.run_node_counts is not None
            else args.node_counts
        )
        run_command_args.extend(
            ["--node-counts"] + [str(n) for n in node_counts_for_run]
        )

        # Rest of the arguments remain the same
        run_command_args.extend(["--graphs-per-size", str(args.graphs_per_size)])
        run_command_args.extend(["--log-dir", args.log_dir])

        # Add checkpoint info
        run_command_args.extend(["--timestamp", timestamp])
        if global_checkpoint or args.graphs_checkpoint:
            run_command_args.extend(
                ["--graphs-checkpoint", args.graphs_checkpoint or global_checkpoint]
            )

        exit_code = run_command(run_command_args, logger)
        if exit_code != 0:
            logger.error("Experiments failed. Exiting.")
            return exit_code

    if args.analyze or args.all:
        logger.info("=" * 50)
        logger.info("STEP 3: Analyzing results")
        logger.info("=" * 50)

        # Build command with optional results file
        analyze_command_args = [sys.executable, analyze_script]
        if args.results:
            analyze_command_args.extend(["--results", args.results])

        # Add visualization parameters
        analyze_command_args.extend(
            ["--log-scale" if args.log_scale else "--no-log-scale"]
        )
        analyze_command_args.extend(
            [
                "--metrics",
                "path_cost",
                "path_length",
                "nodes_expanded",
                "runtime_ms",
                "peak_memory_kb",
            ]
        )
        analyze_command_args.extend(["--log-dir", args.log_dir])

        # Add checkpoint info
        analyze_command_args.extend(["--timestamp", timestamp])
        if global_checkpoint or args.results_checkpoint:
            analyze_command_args.extend(
                ["--results-checkpoint", args.results_checkpoint or global_checkpoint]
            )

        exit_code = run_command(analyze_command_args, logger)
        if exit_code != 0:
            logger.error("Analysis failed. Exiting.")
            return exit_code

    elapsed_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"Experiment completed in {elapsed_time:.2f} seconds")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())

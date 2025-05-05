"""
SCATS Traffic Prediction - Utilities Module
This module contains utility functions for data processing, visualization, analysis and search algorithms.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

import heapq
import math


def distance(a, b):
    """The euclid distance between two (x, y) points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. Efficiently supports item lookup and updates."""

    def __init__(self, order="min", f=lambda x: x):
        self.heap = []
        # Map from items to their heap index and priority value
        self.entry_finder = {}
        self.counter = 0  # Unique sequence count for tiebreaking
        if order == "min":
            self.f = f
        elif order == "max":
            self.f = lambda x: -f(x)
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        # If item already in queue, remove it first
        if item in self.entry_finder:
            self.remove_item(item)

        # Get priority and add entry counter for stable ordering
        priority = self.f(item)
        count = self.counter
        self.counter += 1

        # Add to heap and remember entry
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.heap, entry)

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item with lowest f(x) value."""
        while self.heap:
            priority, count, item = heapq.heappop(self.heap)
            if item in self.entry_finder:
                del self.entry_finder[item]
                return item
        raise Exception("Trying to pop from empty PriorityQueue.")

    def remove_item(self, item):
        """Mark an existing item as removed. Raises KeyError if not found."""
        if item in self.entry_finder:
            entry = self.entry_finder[item]
            # Mark as removed by pointing to None
            entry[-1] = None
            del self.entry_finder[item]
        else:
            raise KeyError(f"{item} not in priority queue")

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.entry_finder)

    def __contains__(self, item):
        """Return True if the key is in PriorityQueue."""
        return item in self.entry_finder

    def __getitem__(self, item):
        """Returns the priority value associated with item."""
        if item in self.entry_finder:
            return self.entry_finder[item][0]
        raise KeyError(f"{item} not in priority queue")

    def __delitem__(self, item):
        """Remove item from queue."""
        self.remove_item(item)


def create_directory(directory):
    """
    Create a directory if it doesn't exist

    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def save_dict_to_json(data, filepath):
    """
    Save a dictionary to a JSON file

    Args:
        data: Dictionary to save
        filepath: Path to save the JSON file
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4, default=str)


def load_json_to_dict(filepath):
    """
    Load a JSON file to a dictionary

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary with the loaded data
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def plot_traffic_heatmap(df, location, output_dir="figures"):
    """
    Create a heatmap of traffic patterns for a specific location

    Args:
        df: DataFrame with traffic data
        location: Location to plot (using Location field instead of SCATS Number)
        output_dir: Directory to save the plot
    """
    # Create output directory if it doesn't exist
    create_directory(output_dir)

    # Filter data for the specified location
    location_data = df[df["Location"] == location].copy()

    # If no data found, try looking for a partial match
    if len(location_data) == 0:
        matching_locations = [
            loc for loc in df["Location"].unique() if location.lower() in loc.lower()
        ]
        if matching_locations:
            location = matching_locations[0]
            location_data = df[df["Location"] == location].copy()
            print(f"No exact match found, using similar location: {location}")
        else:
            print(f"No data found for location: {location}")
            return

    # Convert Date to datetime if it's a string
    if not pd.api.types.is_datetime64_any_dtype(location_data["Date"]):
        location_data["Date"] = pd.to_datetime(location_data["Date"])

    # Extract day of week and hour
    location_data["day_of_week"] = location_data["Date"].dt.dayofweek
    location_data["hour"] = location_data["interval_id"] // 4

    # Create pivot table for heatmap
    pivot_data = location_data.pivot_table(
        values="traffic_volume", index="day_of_week", columns="hour", aggfunc="mean"
    )

    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        pivot_data,
        cmap="YlGnBu",
        annot=False,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "Average Traffic Volume"},
    )

    # Set labels and title
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.title(f"Traffic Pattern Heatmap for {location}")

    # Customize ticks
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    plt.yticks(np.arange(0.5, 7.5), day_names)
    plt.xticks(np.arange(0.5, 24.5, 1), range(0, 24))

    # Create safe filename
    safe_location = location.replace(" ", "_").replace("/", "_")

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"heatmap_{safe_location}.png"), dpi=300)
    plt.close()


def analyze_model_performance(metrics_file):
    """
    Analyze model performance metrics

    Args:
        metrics_file: Path to the metrics JSON file

    Returns:
        Analysis results
    """
    # Load metrics
    metrics = load_json_to_dict(metrics_file)

    # Extract overall metrics
    overall_metrics = metrics["overall_metrics"]
    print(f"Overall MSE: {overall_metrics['mse']:.4f}")
    print(f"Overall MAE: {overall_metrics['mae']:.4f}")
    print(f"Overall RMSE: {overall_metrics['rmse']:.4f}")

    # Analyze location-specific performance
    location_metrics = metrics["location_metrics"]
    location_rmse = {loc: data["rmse"] for loc, data in location_metrics.items()}

    # Find best and worst performing locations
    best_locations = sorted(location_rmse.items(), key=lambda x: x[1])[:5]
    worst_locations = sorted(location_rmse.items(), key=lambda x: x[1], reverse=True)[
        :5
    ]

    print("\nBest performing locations (lowest RMSE):")
    for loc, rmse in best_locations:
        scats_number = location_metrics[loc].get("scats_number", "N/A")
        print(f"Location: {loc} (SCATS {scats_number}): RMSE = {rmse:.4f}")

    print("\nWorst performing locations (highest RMSE):")
    for loc, rmse in worst_locations:
        scats_number = location_metrics[loc].get("scats_number", "N/A")
        print(f"Location: {loc} (SCATS {scats_number}): RMSE = {rmse:.4f}")

    # Analyze time-of-day performance
    hour_metrics = metrics["hour_metrics"]
    hour_rmse = {int(hour): data["rmse"] for hour, data in hour_metrics.items()}

    # Find best and worst performing hours
    best_hours = sorted(hour_rmse.items(), key=lambda x: x[1])[:3]
    worst_hours = sorted(hour_rmse.items(), key=lambda x: x[1], reverse=True)[:3]

    print("\nBest performing hours (lowest RMSE):")
    for hour, rmse in best_hours:
        print(f"{hour:02d}:00: RMSE = {rmse:.4f}")

    print("\nWorst performing hours (highest RMSE):")
    for hour, rmse in worst_hours:
        print(f"{hour:02d}:00: RMSE = {rmse:.4f}")

    # Analyze day-of-week performance
    dow_metrics = metrics["day_of_week_metrics"]
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    dow_rmse = {day_names[int(dow)]: data["rmse"] for dow, data in dow_metrics.items()}

    print("\nPerformance by day of week (RMSE):")
    for day, rmse in dow_rmse.items():
        print(f"{day}: RMSE = {rmse:.4f}")

    # Analyze weekend vs weekday performance if available
    if "weekend_metrics" in metrics and metrics["weekend_metrics"]:
        weekend_metrics = metrics["weekend_metrics"]
        print("\nWeekend vs Weekday performance:")
        print(
            f"Weekend RMSE: {weekend_metrics['weekend_rmse']:.4f} (samples: {weekend_metrics['weekend_samples']})"
        )
        print(
            f"Weekday RMSE: {weekend_metrics['weekday_rmse']:.4f} (samples: {weekend_metrics['weekday_samples']})"
        )

    # Return analysis results
    return {
        "overall_metrics": overall_metrics,
        "best_locations": best_locations,
        "worst_locations": worst_locations,
        "best_hours": best_hours,
        "worst_hours": worst_hours,
        "dow_performance": dow_rmse,
    }


def compare_models(metrics_files, model_names=None):
    """
    Compare performance of multiple models

    Args:
        metrics_files: List of paths to metrics JSON files
        model_names: List of model names (optional)

    Returns:
        Comparison results
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(metrics_files))]

    # Load metrics for each model
    all_metrics = []
    for file in metrics_files:
        all_metrics.append(load_json_to_dict(file))

    # Extract overall metrics
    overall_metrics = []
    for metrics in all_metrics:
        overall_metrics.append(metrics["overall_metrics"])

    # Create comparison table
    comparison = pd.DataFrame(overall_metrics, index=model_names)

    print("Model Comparison:")
    print(comparison)

    # Create bar chart
    plt.figure(figsize=(10, 6))
    comparison[["rmse", "mae"]].plot(kind="bar", alpha=0.7)
    plt.title("Model Performance Comparison")
    plt.ylabel("Error")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300)
    plt.close()

    return comparison


def plot_prediction_intervals(
    predictions,
    timestamps,
    location,
    percentiles=[5, 25, 75, 95],
    output_dir="predictions",
):
    """
    Plot predictions with uncertainty intervals

    Args:
        predictions: List of predictions
        timestamps: List of timestamps
        location: Location name
        percentiles: List of percentiles for confidence intervals
        output_dir: Directory to save the plot
    """
    # Create output directory if it doesn't exist
    create_directory(output_dir)

    # Convert timestamps to datetime
    dt_timestamps = pd.to_datetime(timestamps)

    # Create DataFrame
    df = pd.DataFrame({"timestamp": dt_timestamps, "prediction": predictions})

    # Extract hour of day
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Calculate prediction intervals by hour
    intervals = {}
    for p in percentiles:
        intervals[f"p{p}"] = df.groupby("hour")["prediction"].quantile(p / 100)

    # Calculate mean prediction by hour
    mean_by_hour = df.groupby("hour")["prediction"].mean()

    # Create safe filename
    safe_location = location.replace(" ", "_").replace("/", "_")

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot mean prediction
    plt.plot(mean_by_hour.index, mean_by_hour.values, "b-", linewidth=2, label="Mean")

    # Plot intervals
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(percentiles) // 2))
    for i in range(0, len(percentiles), 2):
        if i + 1 < len(percentiles):
            lower_p = percentiles[i]
            upper_p = percentiles[i + 1]
            plt.fill_between(
                mean_by_hour.index,
                intervals[f"p{lower_p}"].values,
                intervals[f"p{upper_p}"].values,
                alpha=0.3,
                color=colors[i // 2],
                label=f"{lower_p}th-{upper_p}th percentile",
            )

    plt.title(f"Traffic Prediction with Confidence Intervals for {location}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Traffic Volume")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"prediction_intervals_{safe_location}.png"), dpi=300
    )
    plt.close()


def analyze_prediction_patterns(predictions, timestamps, output_dir="analysis"):
    """
    Analyze patterns in predictions

    Args:
        predictions: Dictionary with predictions for each location
        timestamps: Dictionary with timestamps for each location
        output_dir: Directory to save analysis results

    Returns:
        Analysis results
    """
    # Create output directory if it doesn't exist
    create_directory(output_dir)

    # Initialize results
    results = {}

    # Process each location
    for location, pred_data in predictions.items():
        # Extract predictions and timestamps
        preds = pred_data["predictions"]
        times = pd.to_datetime(pred_data["timestamps"])

        # Create DataFrame
        df = pd.DataFrame({"timestamp": times, "prediction": preds})

        # Extract temporal components
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["date"] = df["timestamp"].dt.date
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Calculate daily statistics
        daily_stats = (
            df.groupby("date")["prediction"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )

        # Calculate hourly pattern
        hourly_pattern = df.groupby("hour")["prediction"].mean()

        # Calculate day of week pattern
        dow_pattern = df.groupby("day_of_week")["prediction"].mean()

        # Calculate weekday/weekend pattern
        weekend_pattern = df.groupby("is_weekend")["prediction"].mean()
        weekend_hourly = (
            df.groupby(["is_weekend", "hour"])["prediction"].mean().unstack()
        )

        # Store results
        results[location] = {
            "daily_stats": daily_stats.to_dict(orient="records"),
            "hourly_pattern": hourly_pattern.to_dict(),
            "dow_pattern": dow_pattern.to_dict(),
            "weekend_pattern": weekend_pattern.to_dict(),
            "weekend_hourly_pattern": (
                weekend_hourly.to_dict() if not weekend_hourly.empty else {}
            ),
        }

    # Save results
    save_dict_to_json(results, os.path.join(output_dir, "prediction_patterns.json"))

    return results


def format_time_for_display(interval_id):
    """
    Format interval ID as HH:MM time string

    Args:
        interval_id: Interval ID (0-95)

    Returns:
        Formatted time string
    """
    hours = interval_id // 4
    minutes = (interval_id % 4) * 15
    return f"{hours:02d}:{minutes:02d}"


def calculate_weighted_mape(y_true, y_pred, weights=None):
    """
    Calculate weighted Mean Absolute Percentage Error

    Args:
        y_true: Actual values
        y_pred: Predicted values
        weights: Weights for each sample (optional)

    Returns:
        Weighted MAPE value
    """
    # Handle division by zero
    mask = y_true != 0
    y_true_safe = y_true[mask]
    y_pred_safe = y_pred[mask]

    # Calculate absolute percentage error
    ape = np.abs((y_true_safe - y_pred_safe) / y_true_safe) * 100

    if weights is None:
        # Unweighted MAPE
        return np.mean(ape)
    else:
        # Ensure weights match the filtered data
        weights_safe = weights[mask]

        # Normalize weights
        weights_safe = weights_safe / np.sum(weights_safe)

        # Calculate weighted MAPE
        return np.sum(weights_safe * ape)


def compare_approaches_for_intersection(df, scats_number, output_dir="figures"):
    """
    Compare traffic patterns at different approaches to the same intersection

    Args:
        df: DataFrame with traffic data
        scats_number: SCATS Number (intersection ID) to analyze
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    create_directory(output_dir)

    # Filter data for the specified SCATS Number
    intersection_data = df[df["SCATS Number"] == scats_number].copy()

    # If no data found, return
    if len(intersection_data) == 0:
        print(f"No data found for SCATS Number: {scats_number}")
        return

    # Get unique locations (approaches) for this intersection
    approaches = intersection_data["Location"].unique()

    # Convert Date to datetime if it's a string
    if not pd.api.types.is_datetime64_any_dtype(intersection_data["Date"]):
        intersection_data["Date"] = pd.to_datetime(intersection_data["Date"])

    # Extract hour from interval_id
    intersection_data["hour"] = intersection_data["interval_id"] // 4

    # Create daily pattern comparison
    plt.figure(figsize=(14, 8))

    for approach in approaches:
        # Get data for this approach
        approach_data = intersection_data[intersection_data["Location"] == approach]

        # Calculate hourly average
        hourly_avg = approach_data.groupby("hour")["traffic_volume"].mean()

        # Plot hourly average
        plt.plot(hourly_avg.index, hourly_avg.values, marker="o", label=approach)

    plt.title(f"Daily Traffic Pattern Comparison for Intersection {scats_number}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Traffic Volume")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))
    plt.legend(title="Approach", loc="best")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"approach_comparison_{scats_number}.png"), dpi=300
    )
    plt.close()

    # Create weekly pattern comparison
    # Extract day of week
    intersection_data["day_of_week"] = intersection_data["Date"].dt.dayofweek

    plt.figure(figsize=(14, 8))

    for approach in approaches:
        # Get data for this approach
        approach_data = intersection_data[intersection_data["Location"] == approach]

        # Calculate daily average
        daily_avg = approach_data.groupby("day_of_week")["traffic_volume"].mean()

        # Plot daily average
        day_indices = [0, 1, 2, 3, 4, 5, 6]
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        plt.plot(
            day_indices,
            [daily_avg.get(d, 0) for d in day_indices],
            marker="o",
            label=approach,
        )

    plt.title(f"Weekly Traffic Pattern Comparison for Intersection {scats_number}")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Traffic Volume")
    plt.grid(True, alpha=0.3)
    plt.xticks(day_indices, day_names)
    plt.legend(title="Approach", loc="best")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"weekly_comparison_{scats_number}.png"), dpi=300
    )
    plt.close()

    # Create heatmap for each approach
    for approach in approaches:
        # Create a safe filename
        safe_approach = approach.replace(" ", "_").replace("/", "_")

        # Filter data for this approach
        approach_data = intersection_data[intersection_data["Location"] == approach]

        # Create pivot table for heatmap
        pivot_data = approach_data.pivot_table(
            values="traffic_volume", index="day_of_week", columns="hour", aggfunc="mean"
        )

        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            pivot_data,
            cmap="YlGnBu",
            annot=False,
            fmt=".0f",
            linewidths=0.5,
            cbar_kws={"label": "Average Traffic Volume"},
        )

        # Set labels and title
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week")
        plt.title(f"Traffic Pattern Heatmap for {approach}")

        # Customize ticks
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        plt.yticks(np.arange(0.5, 7.5), day_names)
        plt.xticks(np.arange(0.5, 24.5, 1), range(0, 24))

        # Save the plot
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"heatmap_{scats_number}_{safe_approach}.png"),
            dpi=300,
        )
        plt.close()


if __name__ == "__main__":
    # Example usage
    analyze_model_performance("trained_model/evaluation_metrics.json")

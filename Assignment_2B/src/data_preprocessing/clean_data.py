"""
SCATS Traffic Prediction - Data Processing Module
This module handles loading, reshaping, and cleaning the SCATS traffic data.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
from src.utils.utils import plot_traffic_heatmap


def load_and_prepare_data(file_path):
    """
    Load the SCATS traffic data and perform initial preparations

    Args:
        file_path: Path to the SCATS CSV file

    Returns:
        DataFrame with prepared data and volume columns
    """
    print(f"Loading data from {file_path}...")

    # Load the data
    df = pd.read_csv(file_path)

    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # Check data dimensions
    print(f"Data dimensions: {df.shape}")
    print(
        f"Number of unique SCATS Numbers (intersections): {df['SCATS Number'].nunique()}"
    )
    print(f"Number of unique Locations (approaches): {df['Location'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Identify traffic volume columns (V00-V95)
    volume_cols = [
        col for col in df.columns if col.startswith("V") and col[1:].isdigit()
    ]

    # Basic data validation
    print(f"Number of traffic volume columns: {len(volume_cols)}")

    return df, volume_cols


def reshape_data(df, volume_cols):
    """
    Reshape the data from wide format (one row per day) to long format (one row per interval)

    Args:
        df: DataFrame with SCATS data
        volume_cols: List of volume column names

    Returns:
        Reshaped DataFrame
    """
    print("Reshaping data to long format...")

    # Create a list to hold the transformed data
    rows_list = []

    # Process each row in the original dataframe
    for _, row in df.iterrows():
        scats_number = row["SCATS Number"]
        location = row["Location"]
        date = row["Date"]
        lat = row["NB_LATITUDE"]
        lon = row["NB_LONGITUDE"]

        # For each volume column, create a new row
        for i, col in enumerate(volume_cols):
            # Calculate the time of day for this interval
            interval_id = i
            hours = i // 4
            minutes = (i % 4) * 15
            time_of_day = f"{hours:02d}:{minutes:02d}"

            # Create a new row
            new_row = {
                "SCATS Number": scats_number,
                "Location": location,
                "Date": date,
                "NB_LATITUDE": lat,
                "NB_LONGITUDE": lon,
                "interval_id": interval_id,
                "time_of_day": time_of_day,
                "traffic_volume": row[col],
            }

            rows_list.append(new_row)

    # Create a new dataframe
    long_df = pd.DataFrame(rows_list)

    print(f"Reshaped data dimensions: {long_df.shape}")

    return long_df


def analyze_scats_to_location_relationship(df, visualize_output_dir="figures"):
    """
    Analyze the relationship between SCATS Numbers and Locations

    Args:
        df: DataFrame with SCATS data

    Returns:
        Dictionary with analysis results
    """
    print("Analyzing SCATS Number to Location relationship...")

    # Create a mapping from SCATS Number to Locations
    scats_to_locations = {}

    for scats_number, group in df.groupby("SCATS Number"):
        locations = group["Location"].unique()
        scats_to_locations[scats_number] = list(locations)

    # Count locations per SCATS Number
    locations_per_scats = {
        scats: len(locs) for scats, locs in scats_to_locations.items()
    }

    # Calculate summary statistics
    avg_locations = np.mean(list(locations_per_scats.values()))
    max_locations = max(locations_per_scats.values())
    min_locations = min(locations_per_scats.values())

    print(f"Average locations per SCATS Number: {avg_locations:.2f}")
    print(f"Maximum locations per SCATS Number: {max_locations}")
    print(f"Minimum locations per SCATS Number: {min_locations}")

    # Find SCATS Numbers with the most locations
    top_scats = sorted(locations_per_scats.items(), key=lambda x: x[1], reverse=True)[
        :5
    ]
    print("\nSCATS Numbers with the most locations:")
    for scats, count in top_scats:
        print(f"SCATS {scats}: {count} locations - {scats_to_locations[scats]}")

    # Create a histogram of locations per SCATS Number
    counts = list(locations_per_scats.values())
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=range(1, max(counts) + 2), alpha=0.7)
    plt.title("Distribution of Locations per SCATS Number")
    plt.xlabel("Number of Locations")
    plt.ylabel("Number of SCATS Numbers")
    plt.grid(True, alpha=0.3, axis="y")

    save_path = os.path.join(visualize_output_dir, "locations_per_scats_histogram.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {
        "scats_to_locations": scats_to_locations,
        "summary": {
            "avg_locations": avg_locations,
            "max_locations": max_locations,
            "min_locations": min_locations,
            "top_scats": top_scats,
        },
    }


def clean_problematic_locations(df):
    """
    Identify and remove locations with data issues

    Args:
        df: DataFrame with SCATS data

    Returns:
        Cleaned DataFrame
    """
    print("Identifying and cleaning problematic locations...")

    # Count records per location and date
    location_date_counts = (
        df.groupby(["Location", "Date"]).size().reset_index(name="count")
    )

    # Find locations with duplicate entries (more than 96 intervals per day)
    duplicate_locations = location_date_counts[location_date_counts["count"] > 96][
        "Location"
    ].unique()
    print(f"Locations with duplicate entries: {len(duplicate_locations)}")

    # Count days per location
    days_per_location = df.groupby("Location")["Date"].nunique()

    # Find locations with very few days of data (less than 5 days)
    sparse_locations = days_per_location[days_per_location < 5].index
    print(f"Locations with less than 5 days of data: {len(sparse_locations)}")

    # Create a list of all problematic locations
    problematic_locations = list(set(duplicate_locations) | set(sparse_locations))
    print(f"Total problematic locations to remove: {len(problematic_locations)}")

    # Remove problematic locations
    df_clean = df[~df["Location"].isin(problematic_locations)].copy()

    # Print statistics after cleaning
    print(f"Data dimensions after cleaning: {df_clean.shape}")
    print(
        f"Removed {len(df) - len(df_clean)} rows ({(len(df) - len(df_clean))/len(df):.1%} of data)"
    )
    print(f"Unique SCATS Numbers after cleaning: {df_clean['SCATS Number'].nunique()}")
    print(f"Unique Locations after cleaning: {df_clean['Location'].nunique()}")

    return df_clean


def analyze_data_patterns(df):
    """
    Analyze patterns in the data for better understanding

    Args:
        df: DataFrame with SCATS data

    Returns:
        Dictionary with analysis results
    """
    print("Analyzing data patterns...")

    # Create a proper copy of the dataframe to avoid SettingWithCopyWarning
    df_analysis = df.copy()

    # Check missing data patterns
    missing_dates = {}

    # Group by location (not SCATS Number)
    for location, location_df in df_analysis.groupby("Location"):
        # Get unique dates for this location
        dates = location_df["Date"].dt.date.unique()
        dates = sorted(dates)

        # Get date range
        min_date = min(dates)
        max_date = max(dates)

        # Create complete date range
        all_dates = pd.date_range(start=min_date, end=max_date, freq="D").date

        # Find missing dates
        missing = set(all_dates) - set(dates)
        if missing:
            missing_dates[location] = sorted(list(missing))

    # Calculate traffic patterns by hour of day - use proper assignment
    df_analysis.loc[:, "hour"] = df_analysis["interval_id"] // 4
    hourly_traffic = (
        df_analysis.groupby("hour")["traffic_volume"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    # Calculate traffic patterns by day of week - use proper assignment
    df_analysis.loc[:, "day_of_week"] = df_analysis["Date"].dt.dayofweek
    daily_traffic = (
        df_analysis.groupby("day_of_week")["traffic_volume"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    daily_traffic["day_name"] = daily_traffic["day_of_week"].map(
        {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
    )

    # Calculate weekend vs weekday patterns - use proper assignment
    df_analysis.loc[:, "is_weekend"] = (df_analysis["day_of_week"] >= 5).astype(int)
    weekend_traffic = (
        df_analysis.groupby("is_weekend")["traffic_volume"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    weekend_traffic["day_type"] = weekend_traffic["is_weekend"].map(
        {0: "Weekday", 1: "Weekend"}
    )

    # Return analysis results
    return {
        "missing_dates": missing_dates,
        "hourly_traffic": hourly_traffic,
        "daily_traffic": daily_traffic,
        "weekend_traffic": weekend_traffic,
    }


def visualize_data_patterns(df, analysis_results, output_dir="figures"):
    """
    Visualize patterns in the data

    Args:
        analysis_results: Dictionary with analysis results
        output_dir: Directory to save figures
    """
    print("Visualizing data patterns...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot hourly traffic patterns
    hourly_traffic = analysis_results["hourly_traffic"]
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_traffic["hour"], hourly_traffic["mean"], "b-", linewidth=2)
    plt.fill_between(
        hourly_traffic["hour"],
        hourly_traffic["mean"] - hourly_traffic["std"],
        hourly_traffic["mean"] + hourly_traffic["std"],
        alpha=0.2,
        color="b",
    )
    plt.title("Average Traffic Volume by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Traffic Volume")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))
    plt.savefig(os.path.join(output_dir, "hourly_traffic_pattern.png"), dpi=300)
    plt.close()

    # Plot daily traffic patterns
    daily_traffic = analysis_results["daily_traffic"]
    plt.figure(figsize=(10, 6))
    plt.bar(
        daily_traffic["day_name"],
        daily_traffic["mean"],
        yerr=daily_traffic["std"],
        alpha=0.7,
    )
    plt.title("Average Traffic Volume by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Traffic Volume")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "daily_traffic_pattern.png"), dpi=300)
    plt.close()

    # Plot weekend vs weekday patterns
    weekend_traffic = analysis_results["weekend_traffic"]
    plt.figure(figsize=(8, 6))
    plt.bar(
        weekend_traffic["day_type"],
        weekend_traffic["mean"],
        yerr=weekend_traffic["std"],
        alpha=0.7,
    )
    plt.title("Average Traffic Volume: Weekday vs Weekend")
    plt.xlabel("Day Type")
    plt.ylabel("Traffic Volume")
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig(os.path.join(output_dir, "weekend_vs_weekday_pattern.png"), dpi=300)
    plt.close()

    # Plot missing dates distribution
    missing_dates = analysis_results["missing_dates"]
    missing_counts = [len(dates) for dates in missing_dates.values()]
    plt.figure(figsize=(10, 6))
    plt.hist(missing_counts, bins=range(1, max(missing_counts) + 2), alpha=0.7)
    plt.title("Distribution of Missing Days per Location")
    plt.xlabel("Number of Missing Days")
    plt.ylabel("Number of Locations")
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig(os.path.join(output_dir, "missing_days_distribution.png"), dpi=300)
    plt.close()

    # Create a heatmap of traffic by hour and day of week
    # First create a sample of the data (for a few intersections)
    sample_scats = np.random.choice(
        list(missing_dates.keys()), min(5, len(missing_dates)), replace=False
    )

    # For each sampled location, create a heatmap
    for location in sample_scats:
        # Create a safe filename
        safe_location = location.replace(" ", "_").replace("/", "_")

        # Create a heatmap of hourly patterns by day of week

        plot_traffic_heatmap(df, location, output_dir)


def visualize_intersection_approaches(df, output_dir="figures"):
    """
    Visualize traffic patterns for different approaches to the same intersection

    Args:
        df: DataFrame with SCATS data
        output_dir: Directory to save figures
    """
    print("Visualizing intersection approaches...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find intersections with multiple approaches
    scats_to_locations = {}
    for scats, group in df.groupby("SCATS Number"):
        locations = group["Location"].unique()
        if len(locations) > 1:
            scats_to_locations[scats] = list(locations)

    # Select a sample of intersections to visualize
    sample_scats = list(scats_to_locations.keys())[:5]

    # For each sampled intersection, create comparison plots
    for scats in sample_scats:
        # Get locations for this intersection
        locations = scats_to_locations[scats]

        # Filter data for this intersection
        scats_data = df[df["SCATS Number"] == scats].copy()

        # Calculate hourly averages for each location
        hourly_averages = {}
        for location in locations:
            loc_data = scats_data[scats_data["Location"] == location]
            loc_data["hour"] = loc_data["interval_id"] // 4
            hourly_avg = loc_data.groupby("hour")["traffic_volume"].mean()
            hourly_averages[location] = hourly_avg

        # Create plot
        plt.figure(figsize=(14, 8))
        for location, hourly_avg in hourly_averages.items():
            shortened_location = (
                location.split("_")[0] + "..." + location.split("_")[-1]
            )
            plt.plot(
                hourly_avg.index,
                hourly_avg.values,
                marker="o",
                label=shortened_location,
            )

        plt.title(f"Traffic Patterns for Different Approaches to Intersection {scats}")
        plt.xlabel("Hour of Day")
        plt.ylabel("Average Traffic Volume")
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        plt.legend(title="Approach", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"intersection_{scats}_approaches.png"), dpi=300
        )
        plt.close()


def process_data(
    file_path, output_dir="processed_data", visualize_output_dir="figures"
):
    """
    Complete data processing workflow

    Args:
        file_path: Path to the input CSV file
        output_dir: Directory to save processed data

    Returns:
        Processed DataFrame
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visualize_output_dir, exist_ok=True)

    # Load and prepare data
    df, volume_cols = load_and_prepare_data(file_path)

    # Analyze SCATS Number to Location relationship
    scats_location_analysis = analyze_scats_to_location_relationship(
        df, visualize_output_dir
    )

    # Reshape data
    df_long = reshape_data(df, volume_cols)

    # Visualize intersection approaches
    visualize_intersection_approaches(df_long, visualize_output_dir)

    # Clean problematic locations
    df_clean = clean_problematic_locations(df_long)

    # Analyze data patterns
    analysis_results = analyze_data_patterns(df_clean)

    # Visualize data patterns
    visualize_data_patterns(df_clean, analysis_results, visualize_output_dir)

    # Save processed data
    df_clean.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)

    print("Data processing completed successfully!")

    return df_clean


if __name__ == "__main__":
    # Process data
    df = process_data(
        file_path="data_preprocessing/raw_data/Scats Data October 2006.csv",
        output_dir="processed_data/preprocessed_data",
        visualize_output_dir="data_preprocessing/eda_insights",
    )

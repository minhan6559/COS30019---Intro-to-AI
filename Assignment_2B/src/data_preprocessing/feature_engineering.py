"""
SCATS Traffic Prediction - Feature Engineering Module
This module handles feature creation, sequence generation, and train-test splitting.
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def engineer_features(df):
    """
    Engineer features for time series prediction that are effective yet easy to re-engineer

    Args:
        df: DataFrame with SCATS data

    Returns:
        DataFrame with engineered features and location mapping dictionary
    """
    print("Engineering features...")

    # Copy the dataframe to avoid modifying the original
    df_features = df.copy()

    # Sort by location, date, and interval for proper lag feature creation
    df_features = df_features.sort_values(["Location", "Date", "interval_id"])

    # 1. Temporal features
    # Day of week (0 = Monday, 6 = Sunday)
    df_features["day_of_week"] = df_features["Date"].dt.dayofweek

    # Add is_weekend feature
    df_features["is_weekend"] = (df_features["day_of_week"] >= 5).astype(int)

    # Sinusoidal encoding of day of week
    df_features["dow_sin"] = np.sin(df_features["day_of_week"] * (2 * np.pi / 7))
    df_features["dow_cos"] = np.cos(df_features["day_of_week"] * (2 * np.pi / 7))

    # Sinusoidal encoding of time of day
    df_features["tod_sin"] = np.sin(df_features["interval_id"] * (2 * np.pi / 96))
    df_features["tod_cos"] = np.cos(df_features["interval_id"] * (2 * np.pi / 96))

    # 2. Gap handling features
    # Calculate days since previous observation (for each location)
    df_features["days_since_prev"] = (
        df_features.groupby("Location")["Date"].diff().dt.days.fillna(0)
    )

    # Boolean flag for data after a gap
    df_features["after_gap"] = (df_features["days_since_prev"] > 1).astype(int)

    # 3. Create location encodings
    # Get unique locations - Using Location field instead of SCATS Number
    locations = df_features["Location"].unique()
    location_to_idx = {loc: idx for idx, loc in enumerate(locations)}

    # Add location index
    df_features["location_idx"] = df_features["Location"].map(location_to_idx)

    # 4. Create lag features (grouped by location)
    # Create lag features within each location group - Using Location field now
    grouped = df_features.groupby("Location")

    # Previous interval (15 min ago)
    df_features["traffic_lag_1"] = grouped["traffic_volume"].shift(1)

    # Previous hour (4 intervals ago)
    df_features["traffic_lag_4"] = grouped["traffic_volume"].shift(4)

    # Same time yesterday (96 intervals ago)
    df_features["traffic_lag_96"] = grouped["traffic_volume"].shift(96)

    # Average traffic at this time of day for this location
    # This captures the typical pattern for each location and interval
    time_means = df_features.groupby(["Location", "interval_id"])[
        "traffic_volume"
    ].transform("mean")
    df_features["avg_traffic_this_timeofday"] = time_means

    # Fill missing lag values with meaningful defaults
    lag_cols = ["traffic_lag_1", "traffic_lag_4", "traffic_lag_96"]
    for col in lag_cols:
        # Fill with the average traffic at this time of day for this location
        df_features[col].fillna(df_features["avg_traffic_this_timeofday"], inplace=True)

    return df_features, location_to_idx


def create_sequences(df, seq_length=24):
    """
    Create sequences for time series prediction with gap awareness

    Args:
        df: DataFrame with engineered features
        seq_length: Length of input sequences

    Returns:
        X, y, metadata, feature_cols: Sequences, targets, metadata, and feature column names
    """
    print(f"Creating sequences with length {seq_length}...")

    # Define feature columns to use (all engineered features except for non-feature columns)
    feature_cols = [
        "traffic_volume",
        "dow_sin",
        "dow_cos",
        "tod_sin",
        "tod_cos",
        "is_weekend",
        "after_gap",
        "days_since_prev",
        "location_idx",
        "traffic_lag_1",
        "traffic_lag_4",
        "traffic_lag_96",
        "avg_traffic_this_timeofday",
    ]

    # Lists to store sequences and targets
    X_sequences = []
    y_targets = []
    metadata = []  # Store metadata about each sequence

    # Group by location - using Location field now
    location_groups = df.groupby("Location")

    # Process each location
    for location, location_df in location_groups:
        # Sort by date and interval
        location_df = location_df.sort_values(["Date", "interval_id"])

        # Identify continuous segments (break at gaps larger than 1 day)
        segment_id = (location_df["days_since_prev"] > 1).cumsum()

        # Process each continuous segment
        for segment, segment_df in location_df.groupby(segment_id):
            # Skip segments that are too short
            if len(segment_df) < seq_length + 1:
                continue

            # Create sequences using sliding window
            for i in range(len(segment_df) - seq_length):
                # Input sequence
                X_seq = segment_df.iloc[i : i + seq_length][feature_cols].values

                # Target value (next traffic volume)
                y_target = segment_df.iloc[i + seq_length]["traffic_volume"]

                # Store sequence and target
                X_sequences.append(X_seq)
                y_targets.append(y_target)

                # Store metadata
                meta = {
                    "SCATS Number": segment_df.iloc[i]["SCATS Number"],
                    "Location": location,
                    "location_idx": segment_df.iloc[i]["location_idx"],
                    "target_date": segment_df.iloc[i + seq_length]["Date"],
                    "target_interval": segment_df.iloc[i + seq_length]["interval_id"],
                    "target_time": segment_df.iloc[i + seq_length]["time_of_day"],
                }
                metadata.append(meta)

    # Convert lists to arrays
    X = np.array(X_sequences)
    y = np.array(y_targets)

    # Convert metadata to DataFrame
    metadata_df = pd.DataFrame(metadata)

    print(f"Created {len(X)} sequences")

    return X, y, metadata_df, feature_cols


def sequence_based_split(X, y, metadata_df, test_ratio=0.2):
    """
    Split data into training and testing sets based on target dates

    Args:
        X: Input sequences
        y: Target values
        metadata_df: Metadata about each sequence
        test_ratio: Proportion of data to use for testing

    Returns:
        X_train, X_test, y_train, y_test, meta_train, meta_test: Split data
    """
    print(f"Performing sequence-based split with test_ratio={test_ratio}...")

    # Get the temporal split date (last test_ratio of unique dates)
    unique_dates = pd.to_datetime(metadata_df["target_date"]).dt.date.unique()
    unique_dates = np.sort(unique_dates)
    split_idx = int(len(unique_dates) * (1 - test_ratio))
    split_date = unique_dates[split_idx]

    print(f"Split date: {split_date}")

    # Convert metadata target_date to datetime.date objects for comparison
    metadata_df["target_date_obj"] = pd.to_datetime(metadata_df["target_date"]).dt.date

    # Create the split
    train_mask = metadata_df["target_date_obj"] < split_date
    test_mask = ~train_mask

    # Split the data
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Split the metadata
    meta_train = metadata_df[train_mask].reset_index(drop=True)
    meta_test = metadata_df[test_mask].reset_index(drop=True)

    # Print statistics
    print(
        f"Training sequences: {len(X_train)} ({len(X_train)/(len(X_train)+len(X_test)):.1%})"
    )
    print(
        f"Testing sequences: {len(X_test)} ({len(X_test)/(len(X_train)+len(X_test)):.1%})"
    )

    # Check for location coverage
    train_locations = set(meta_train["Location"])
    test_locations = set(meta_test["Location"])

    missing_in_train = test_locations - train_locations
    missing_in_test = train_locations - test_locations

    if missing_in_train:
        print(f"Warning: {len(missing_in_train)} locations have no training data")
    if missing_in_test:
        print(f"Warning: {len(missing_in_test)} locations have no test data")

    return X_train, X_test, y_train, y_test, meta_train, meta_test


def normalize_data(X_train, X_test):
    """
    Normalize the data using StandardScaler

    Args:
        X_train: Training data
        X_test: Testing data

    Returns:
        X_train_scaled, X_test_scaled, scaler: Normalized data and scaler
    """
    print("Normalizing data...")

    # Number of features
    n_features = X_train.shape[2]

    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)

    # Initialize and fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)

    # Transform test data
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)

    # Reshape back
    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)

    return X_train_scaled, X_test_scaled, scaler


def prepare_embedding_inputs(X, feature_cols):
    """
    Prepare inputs for model with location embedding

    Args:
        X: Input sequences
        feature_cols: List of feature column names

    Returns:
        List with feature input and location input
    """
    # Find the index of 'location_idx' in feature_cols
    loc_idx = feature_cols.index("location_idx")

    # Extract location indices
    location_input = X[:, :, loc_idx].astype(int)

    # Create feature input without location_idx
    feature_indices = [i for i in range(X.shape[2]) if i != loc_idx]
    feature_input = X[:, :, feature_indices]

    return [feature_input, location_input]


def save_processed_data(processed_data, output_dir="processed_data"):
    """
    Save the processed data to files with compression
    """
    print(f"Saving processed data to {output_dir}...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save numpy arrays with compression
    np.savez_compressed(
        os.path.join(output_dir, "X_train.npz"), data=processed_data["X_train"]
    )
    np.savez_compressed(
        os.path.join(output_dir, "X_test.npz"), data=processed_data["X_test"]
    )
    np.savez_compressed(
        os.path.join(output_dir, "y_train.npz"), data=processed_data["y_train"]
    )
    np.savez_compressed(
        os.path.join(output_dir, "y_test.npz"), data=processed_data["y_test"]
    )

    # Save metadata
    processed_data["meta_train"].to_csv(
        os.path.join(output_dir, "meta_train.csv"), index=False
    )
    processed_data["meta_test"].to_csv(
        os.path.join(output_dir, "meta_test.csv"), index=False
    )

    # Save scaler
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(processed_data["scaler"], f)

    # Save feature columns
    with open(os.path.join(output_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(processed_data["feature_cols"], f)

    # Save location mapping
    with open(os.path.join(output_dir, "location_to_idx.pkl"), "wb") as f:
        pickle.dump(processed_data["location_to_idx"], f)

    print("Saved all processed data successfully!")


def load_processed_data(input_dir="processed_data"):
    """
    Load the processed data from compressed files
    """
    print(f"Loading processed data from {input_dir}...")

    # Load numpy arrays from compressed files
    X_train = np.load(os.path.join(input_dir, "X_train.npz"))["data"]
    X_test = np.load(os.path.join(input_dir, "X_test.npz"))["data"]
    y_train = np.load(os.path.join(input_dir, "y_train.npz"))["data"]
    y_test = np.load(os.path.join(input_dir, "y_test.npz"))["data"]

    # Load metadata
    meta_train = pd.read_csv(os.path.join(input_dir, "meta_train.csv"))
    meta_test = pd.read_csv(os.path.join(input_dir, "meta_test.csv"))

    # Load scaler
    with open(os.path.join(input_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # Load feature columns
    with open(os.path.join(input_dir, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)

    # Load location mapping
    with open(os.path.join(input_dir, "location_to_idx.pkl"), "rb") as f:
        location_to_idx = pickle.load(f)

    # Prepare embedding inputs (regenerate these from the loaded X data)
    X_train_inputs = prepare_embedding_inputs(X_train, feature_cols)
    X_test_inputs = prepare_embedding_inputs(X_test, feature_cols)

    print("Loaded all processed data successfully!")

    # Return complete dictionary with all data items
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_inputs": X_train_inputs,
        "X_test_inputs": X_test_inputs,
        "y_train": y_train,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "location_to_idx": location_to_idx,
        "n_locations": len(location_to_idx),
        "n_features": X_train.shape[2] - 1,  # Excluding location_idx
    }


def run_feature_engineering(
    df, seq_length=24, test_ratio=0.2, output_dir="processed_data"
):
    """
    Run the complete feature engineering pipeline

    Args:
        df: DataFrame with SCATS data
        seq_length: Length of input sequences
        test_ratio: Proportion of data to use for testing
        output_dir: Directory to save processed data

    Returns:
        Dictionary with processed data
    """
    # Engineer features
    df_features, location_to_idx = engineer_features(df)

    # Create sequences
    X, y, metadata_df, feature_cols = create_sequences(df_features, seq_length)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, meta_train, meta_test = sequence_based_split(
        X, y, metadata_df, test_ratio
    )

    # Normalize the data
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)

    # Prepare data for embedding model
    X_train_inputs = prepare_embedding_inputs(X_train_scaled, feature_cols)
    X_test_inputs = prepare_embedding_inputs(X_test_scaled, feature_cols)

    # Create processed data dictionary
    processed_data = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_inputs": X_train_inputs,
        "X_test_inputs": X_test_inputs,
        "y_train": y_train,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "location_to_idx": location_to_idx,
        "n_locations": len(location_to_idx),
        "n_features": X_train_scaled.shape[2] - 1,  # Excluding location_idx
    }

    # Save processed data
    save_processed_data(processed_data, output_dir)

    print("Feature engineering completed successfully!")

    return processed_data


if __name__ == "__main__":
    # Load processed data from data_processing.py
    df = pd.read_csv("processed_data/cleaned_data.csv")

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    output_dir = "processed_data/preprocessed_data"
    # Run feature engineering
    processed_data = run_feature_engineering(df, output_dir=output_dir)

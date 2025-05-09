"""
SCATS Traffic Prediction - Improved Feature Engineering Module
This module handles feature creation, sequence generation, and train-test splitting
with enhanced lag features and proper scaling.
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
    Engineer features for time series prediction with enhanced lag features
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
    df_features["days_since_prev"] = df_features.groupby("Location")["Date"].transform(
        lambda x: x.diff().dt.days.fillna(0)
    )

    # Boolean flag for data after a gap
    df_features["after_gap"] = (df_features["days_since_prev"] > 1).astype(int)

    # 3. Create location encodings
    locations = df_features["Location"].unique()
    location_to_idx = {loc: idx for idx, loc in enumerate(locations)}
    df_features["location_idx"] = df_features["Location"].map(location_to_idx)

    # 4. Create lag features (grouped by location)
    # --- ORIGINAL LAG FEATURES ---

    # Previous hour (4 intervals ago)
    df_features["traffic_lag_4"] = df_features.groupby("Location")[
        "traffic_volume"
    ].transform(lambda x: x.shift(4))

    # Same time yesterday (96 intervals ago)
    df_features["traffic_lag_96"] = df_features.groupby("Location")[
        "traffic_volume"
    ].transform(lambda x: x.shift(96))

    # --- ENHANCED LAG FEATURES ---
    # 2 hours ago (helps capture medium-term patterns)
    df_features["traffic_lag_8"] = df_features.groupby("Location")[
        "traffic_volume"
    ].transform(lambda x: x.shift(8))

    # Rolling statistics (average of last hour of traffic)
    df_features["rolling_mean_4"] = df_features.groupby("Location")[
        "traffic_volume"
    ].transform(lambda x: x.rolling(4).mean().shift(1))

    # Traffic acceleration (rate of change)
    df_features["traffic_acceleration"] = df_features.groupby("Location")[
        "traffic_volume"
    ].transform(lambda x: x.diff().fillna(0))

    # Average traffic at this time of day for this location
    df_features["avg_traffic_this_timeofday"] = df_features.groupby(
        ["Location", "interval_id"]
    )["traffic_volume"].transform("mean")

    # Fill missing values
    # Where rolling_mean_4 is missing, use the current traffic value
    missing_rolling_mask = df_features["rolling_mean_4"].isna()
    df_features.loc[missing_rolling_mask, "rolling_mean_4"] = df_features.loc[
        missing_rolling_mask, "traffic_volume"
    ]

    # Fill missing lag values with meaningful defaults
    # First, create a dictionary to store averages by location and time of day
    avg_by_loc_tod = {}
    for loc in df_features["Location"].unique():
        loc_data = df_features[df_features["Location"] == loc]
        for tod in range(96):  # 96 intervals in a day
            avg_by_loc_tod[(loc, tod)] = loc_data[loc_data["interval_id"] == tod][
                "traffic_volume"
            ].mean()

    # For each lag feature
    for lag in [4, 8, 96]:
        lag_col = f"traffic_lag_{lag}"
        # Find missing values
        missing_mask = df_features[lag_col].isna()

        # For each missing value, impute with the average for the correct lagged interval
        for idx in df_features[missing_mask].index:
            loc = df_features.loc[idx, "Location"]
            current_tod = df_features.loc[idx, "interval_id"]

            # The time of day for the lagged interval
            lagged_tod = (current_tod - lag) % 96

            # Impute with the average for the location and lagged time of day
            df_features.loc[idx, lag_col] = avg_by_loc_tod.get(
                (loc, lagged_tod), df_features.loc[idx, "avg_traffic_this_timeofday"]
            )

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
        "traffic_lag_4",
        "traffic_lag_8",  # Added new lag feature
        "traffic_lag_96",
        "rolling_mean_4",  # Added rolling mean
        "traffic_acceleration",  # Added traffic acceleration
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


def normalize_data(X_train, X_test, feature_cols):
    """
    Normalize the data properly, keeping categorical and binary features unchanged

    Args:
        X_train: Training data
        X_test: Testing data
        feature_cols: List of feature column names

    Returns:
        X_train_scaled, X_test_scaled, scaler_dict: Normalized data and scalers
    """
    print("Normalizing data with type-appropriate scaling...")

    # Define feature types
    categorical_features = ["location_idx"]
    binary_features = ["is_weekend", "after_gap"]
    cyclical_features = ["dow_sin", "dow_cos", "tod_sin", "tod_cos"]

    # Identify indices for each feature type
    categorical_indices = [
        feature_cols.index(f) for f in categorical_features if f in feature_cols
    ]
    binary_indices = [
        feature_cols.index(f) for f in binary_features if f in feature_cols
    ]
    cyclical_indices = [
        feature_cols.index(f) for f in cyclical_features if f in feature_cols
    ]

    # All other indices are assumed to be continuous features that should be scaled
    all_indices = set(range(len(feature_cols)))
    do_not_scale_indices = set(categorical_indices + binary_indices + cyclical_indices)
    scale_indices = list(all_indices - do_not_scale_indices)

    print(f"Features to scale: {[feature_cols[i] for i in scale_indices]}")
    print(f"Features NOT to scale: {[feature_cols[i] for i in do_not_scale_indices]}")

    # Create copies of input arrays to avoid modifying original data
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Dictionary to store scalers for each feature
    scaler_dict = {}

    # Scale each continuous feature individually
    for idx in scale_indices:
        feature_name = feature_cols[idx]
        # Extract this feature from all sequences
        train_feature = X_train[:, :, idx].reshape(-1, 1)
        test_feature = X_test[:, :, idx].reshape(-1, 1)

        # Create and fit scaler
        scaler = StandardScaler()
        train_feature_scaled = scaler.fit_transform(train_feature)
        test_feature_scaled = scaler.transform(test_feature)

        # Store scaler
        scaler_dict[feature_name] = scaler

        # Put scaled features back
        X_train_scaled[:, :, idx] = train_feature_scaled.reshape(
            X_train.shape[0], X_train.shape[1]
        )
        X_test_scaled[:, :, idx] = test_feature_scaled.reshape(
            X_test.shape[0], X_test.shape[1]
        )

    # No scaling for categorical features - ensure they're integers
    for idx in categorical_indices:
        X_train_scaled[:, :, idx] = X_train[:, :, idx].astype(int)
        X_test_scaled[:, :, idx] = X_test[:, :, idx].astype(int)

    return X_train_scaled, X_test_scaled, scaler_dict


def normalize_target_variable(y_train, y_test, scaler_dict=None):
    """
    Normalize target variables for better model training

    Args:
        y_train: Training target values
        y_test: Testing target values
        scaler_dict: Dictionary to store the scaler

    Returns:
        y_train_scaled, y_test_scaled, y_scaler: Scaled targets and scaler
    """
    # Create and fit scaler on training data
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Add to scaler_dict if provided
    if scaler_dict is not None:
        scaler_dict["y_scaler"] = y_scaler

    return y_train_scaled, y_test_scaled, y_scaler


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

    # Extract location indices - ensure they're integers
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

    # Save original y values if they exist
    if "y_original_train" in processed_data:
        np.savez_compressed(
            os.path.join(output_dir, "y_original_train.npz"),
            data=processed_data["y_original_train"],
        )
        np.savez_compressed(
            os.path.join(output_dir, "y_original_test.npz"),
            data=processed_data["y_original_test"],
        )

    # Save metadata
    processed_data["meta_train"].to_csv(
        os.path.join(output_dir, "meta_train.csv"), index=False
    )
    processed_data["meta_test"].to_csv(
        os.path.join(output_dir, "meta_test.csv"), index=False
    )

    # Save scaler dictionary
    with open(os.path.join(output_dir, "scaler_dict.pkl"), "wb") as f:
        pickle.dump(processed_data["scaler_dict"], f)

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

    # Create result dictionary
    result = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    # Load original y values if they exist
    try:
        y_original_train = np.load(os.path.join(input_dir, "y_original_train.npz"))[
            "data"
        ]
        y_original_test = np.load(os.path.join(input_dir, "y_original_test.npz"))[
            "data"
        ]
        result["y_original_train"] = y_original_train
        result["y_original_test"] = y_original_test
    except FileNotFoundError:
        print(
            "Note: Original y values not found (target normalization may not have been used)"
        )

    # Load metadata
    meta_train = pd.read_csv(os.path.join(input_dir, "meta_train.csv"))
    meta_test = pd.read_csv(os.path.join(input_dir, "meta_test.csv"))
    result["meta_train"] = meta_train
    result["meta_test"] = meta_test

    # Load scaler dictionary
    with open(os.path.join(input_dir, "scaler_dict.pkl"), "rb") as f:
        scaler_dict = pickle.load(f)
    result["scaler_dict"] = scaler_dict

    # Load feature columns
    with open(os.path.join(input_dir, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    result["feature_cols"] = feature_cols

    # Load location mapping
    with open(os.path.join(input_dir, "location_to_idx.pkl"), "rb") as f:
        location_to_idx = pickle.load(f)
    result["location_to_idx"] = location_to_idx

    # Prepare embedding inputs
    X_train_inputs = prepare_embedding_inputs(X_train, feature_cols)
    X_test_inputs = prepare_embedding_inputs(X_test, feature_cols)
    result["X_train_inputs"] = X_train_inputs
    result["X_test_inputs"] = X_test_inputs

    # Add computed values
    result["n_locations"] = len(location_to_idx)
    result["n_features"] = X_train.shape[2] - 1  # Excluding location_idx

    print("Loaded all processed data successfully!")
    return result


def run_feature_engineering(
    df,
    seq_length=24,
    test_ratio=0.2,
    normalize_target=True,
    output_dir="processed_data",
):
    """
    Run the complete improved feature engineering pipeline

    Args:
        df: DataFrame with SCATS data (already cleaned)
        seq_length: Length of input sequences
        test_ratio: Proportion of data to use for testing
        normalize_target: Whether to normalize the target variable
        output_dir: Directory to save processed data

    Returns:
        Dictionary with processed data
    """
    # Engineer features with enhanced lag features
    df_features, location_to_idx = engineer_features(df)

    # Create sequences
    X, y, metadata_df, feature_cols = create_sequences(df_features, seq_length)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, meta_train, meta_test = sequence_based_split(
        X, y, metadata_df, test_ratio
    )

    # Normalize the data properly, respecting feature types
    X_train_scaled, X_test_scaled, scaler_dict = normalize_data(
        X_train, X_test, feature_cols
    )

    # Initialize processed data dictionary
    processed_data = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "scaler_dict": scaler_dict,
        "feature_cols": feature_cols,
        "location_to_idx": location_to_idx,
    }

    # Normalize target variable if requested
    if normalize_target:
        y_train_scaled, y_test_scaled, _ = normalize_target_variable(
            y_train, y_test, scaler_dict
        )
        processed_data["y_train"] = y_train_scaled
        processed_data["y_test"] = y_test_scaled
        processed_data["y_original_train"] = y_train
        processed_data["y_original_test"] = y_test
    else:
        processed_data["y_train"] = y_train
        processed_data["y_test"] = y_test

    # Prepare data for embedding model
    X_train_inputs = prepare_embedding_inputs(X_train_scaled, feature_cols)
    X_test_inputs = prepare_embedding_inputs(X_test_scaled, feature_cols)

    # Add to processed data dictionary
    processed_data["X_train_inputs"] = X_train_inputs
    processed_data["X_test_inputs"] = X_test_inputs
    processed_data["n_locations"] = len(location_to_idx)
    processed_data["n_features"] = X_train_scaled.shape[2] - 1  # Excluding location_idx

    # Save processed data
    save_processed_data(processed_data, output_dir)

    print("Improved feature engineering completed successfully!")

    return processed_data


if __name__ == "__main__":
    # Load already cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv("processed_data/cleaned_data.csv")
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()  # Ensure date is datetime

    # Run improved feature engineering
    output_dir = "processed_data/improved_preprocessed"
    processed_data = run_feature_engineering(
        df, seq_length=24, test_ratio=0.2, normalize_target=True, output_dir=output_dir
    )

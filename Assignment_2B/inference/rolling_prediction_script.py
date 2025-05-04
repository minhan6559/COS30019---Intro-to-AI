"""
Batch Rolling Prediction Script for SCATS Traffic Data
Optimized for GPU batch processing
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import pickle
import os
from tqdm import tqdm
import argparse
from collections import defaultdict
import time

# If tensorflow version is lower than 2.15, raise an error
if tf.__version__ < "2.15":
    raise ImportError(
        "TensorFlow version 2.15 or higher is required to use trained keras file. Please update your TensorFlow installation."
    )

# Setup argparse for command line arguments
parser = argparse.ArgumentParser(
    description="Batch rolling predictions for SCATS traffic data"
)
parser.add_argument(
    "--model_path", type=str, required=True, help="Path to the model file"
)
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="Name of the model (for output folder)",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/content/drive/MyDrive/processed_data/",
    help="Path to processed data",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="/content/drive/MyDrive/predictions/",
    help="Base output path",
)
parser.add_argument(
    "--seq_length", type=int, default=24, help="Sequence length for prediction"
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="Batch size for predictions"
)

args = parser.parse_args()

# Create model-specific output directory
MODEL_OUTPUT_PATH = os.path.join(args.output_path, args.model_name)
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

print(f"Loading model from: {args.model_path}")
print(f"Model name: {args.model_name}")
print(f"Output will be saved to: {MODEL_OUTPUT_PATH}")

# Load model
model = tf.keras.models.load_model(args.model_path)

# Load processed data components
print("Loading processed data components...")
with open(os.path.join(args.data_path, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(args.data_path, "feature_cols.pkl"), "rb") as f:
    feature_cols = pickle.load(f)

with open(os.path.join(args.data_path, "location_to_idx.pkl"), "rb") as f:
    location_to_idx = pickle.load(f)

# Load the original cleaned data
df_clean = pd.read_csv(os.path.join(args.data_path, "cleaned_data.csv"))
df_clean["Date"] = pd.to_datetime(df_clean["Date"])
# Ensure dates are consistent (no time component)
df_clean["Date"] = df_clean["Date"].dt.normalize()


class SequenceManager:
    """Manages sequences for all locations"""

    def __init__(self, locations, seq_length):
        self.locations = locations
        self.seq_length = seq_length
        self.sequences = {loc: pd.DataFrame() for loc in locations}
        self.avg_traffic = {}

    def initialize_sequences(self, df_clean):
        """Initialize sequences with historical data"""
        for location in self.locations:
            loc_data = df_clean[df_clean["Location"] == location].copy()
            # Sort by date and interval
            loc_data = loc_data.sort_values(["Date", "interval_id"])
            # Store average traffic patterns
            self.avg_traffic[location] = (
                loc_data.groupby("interval_id")["traffic_volume"].mean().to_dict()
            )
            # Initialize sequence with available data
            self.sequences[location] = loc_data

    def get_last_sequence(self, location, target_date, target_interval):
        """Get the last seq_length intervals before the target"""
        # Create target datetime
        target_dt = target_date + timedelta(
            hours=int(target_interval // 4), minutes=int((target_interval % 4) * 15)
        )

        # Filter data before target
        loc_seq = self.sequences[location]
        loc_seq = loc_seq.copy()
        loc_seq["datetime"] = loc_seq["Date"] + pd.to_timedelta(
            loc_seq["interval_id"] * 15, unit="m"
        )

        before_target = loc_seq[loc_seq["datetime"] < target_dt].sort_values("datetime")

        if len(before_target) >= self.seq_length:
            return before_target.tail(self.seq_length).drop("datetime", axis=1)
        else:
            return None

    def update_sequence(self, location, date, interval_id, predicted_value):
        """Update sequence with new prediction"""
        new_row = pd.DataFrame(
            [
                {
                    "Location": location,
                    "Date": date,
                    "interval_id": interval_id,
                    "time_of_day": f"{interval_id//4:02d}:{(interval_id%4)*15:02d}",
                    "traffic_volume": predicted_value,
                }
            ]
        )

        # Append to location's sequence
        self.sequences[location] = pd.concat(
            [self.sequences[location], new_row], ignore_index=True
        )
        self.sequences[location] = self.sequences[location].sort_values(
            ["Date", "interval_id"]
        )


def engineer_features_batch(sequences, location_indices, avg_traffic_dict):
    """Engineer features for a batch of sequences"""
    engineered_features = []

    for seq, loc_idx, avg_traffic in zip(sequences, location_indices, avg_traffic_dict):
        seq = seq.copy()

        # Temporal features
        seq["day_of_week"] = seq["Date"].dt.dayofweek
        seq["is_weekend"] = (seq["day_of_week"] >= 5).astype(int)
        seq["dow_sin"] = np.sin(seq["day_of_week"] * (2 * np.pi / 7))
        seq["dow_cos"] = np.cos(seq["day_of_week"] * (2 * np.pi / 7))
        seq["tod_sin"] = np.sin(seq["interval_id"] * (2 * np.pi / 96))
        seq["tod_cos"] = np.cos(seq["interval_id"] * (2 * np.pi / 96))

        # Location features
        seq["location_idx"] = loc_idx

        # Lag features
        seq["traffic_lag_1"] = seq["traffic_volume"].shift(1)
        seq["traffic_lag_4"] = seq["traffic_volume"].shift(4)
        seq["traffic_lag_96"] = seq["traffic_volume"].shift(96)

        # Average traffic for this interval
        seq["avg_traffic_this_timeofday"] = seq["interval_id"].map(avg_traffic)

        # Fill missing lag values
        seq["traffic_lag_1"] = seq["traffic_lag_1"].fillna(
            seq["avg_traffic_this_timeofday"]
        )
        seq["traffic_lag_4"] = seq["traffic_lag_4"].fillna(
            seq["avg_traffic_this_timeofday"]
        )
        seq["traffic_lag_96"] = seq["traffic_lag_96"].fillna(
            seq["avg_traffic_this_timeofday"]
        )

        # Gap handling features
        seq["days_since_prev"] = 0
        seq["after_gap"] = 0

        engineered_features.append(seq[feature_cols].values)

    return np.array(engineered_features)


def prepare_batch_inputs(features_batch, feature_cols):
    """Prepare batch inputs for model"""
    # Normalize features
    batch_size, seq_length, n_features = features_batch.shape
    features_reshaped = features_batch.reshape(-1, n_features)
    features_scaled = scaler.transform(features_reshaped)
    features_scaled = features_scaled.reshape(batch_size, seq_length, n_features)

    # Separate location index from other features
    loc_idx_position = feature_cols.index("location_idx")
    location_input = features_scaled[:, :, loc_idx_position].astype(int)
    feature_input = np.delete(features_scaled, loc_idx_position, axis=2)

    return [feature_input, location_input]


def identify_october_gaps(df_clean):
    """Identify missing intervals in October"""
    october_start = pd.Timestamp("2006-10-01")
    october_end = pd.Timestamp("2006-10-31")

    # Generate all October intervals
    october_dates = pd.date_range(october_start, october_end, freq="D")
    all_october_intervals = []

    for date in october_dates:
        for interval in range(96):
            for location in df_clean["Location"].unique():
                all_october_intervals.append(
                    {"Location": location, "Date": date, "interval_id": interval}
                )

    df_october_all = pd.DataFrame(all_october_intervals)

    # Find what we actually have
    df_october_actual = df_clean[
        (df_clean["Date"] >= october_start) & (df_clean["Date"] <= october_end)
    ].copy()

    # Create keys for comparison
    df_october_all["key"] = (
        df_october_all["Location"]
        + "_"
        + df_october_all["Date"].astype(str)
        + "_"
        + df_october_all["interval_id"].astype(str)
    )
    df_october_actual["key"] = (
        df_october_actual["Location"]
        + "_"
        + df_october_actual["Date"].astype(str)
        + "_"
        + df_october_actual["interval_id"].astype(str)
    )

    # Find missing
    missing_keys = set(df_october_all["key"]) - set(df_october_actual["key"])
    df_october_missing = df_october_all[df_october_all["key"].isin(missing_keys)].drop(
        "key", axis=1
    )

    return df_october_missing.sort_values(["Date", "interval_id", "Location"])


def batch_predict_interval(
    seq_manager, locations, target_date, target_interval, model, scaler, feature_cols
):
    """Predict for all locations at a specific interval"""
    valid_sequences = []
    valid_locations = []
    valid_loc_indices = []
    valid_avg_traffic = []

    # Gather sequences for all locations
    for location in locations:
        seq = seq_manager.get_last_sequence(location, target_date, target_interval)
        if seq is not None:
            valid_sequences.append(seq)
            valid_locations.append(location)
            valid_loc_indices.append(location_to_idx[location])
            valid_avg_traffic.append(seq_manager.avg_traffic[location])

    if not valid_sequences:
        return []

    # Engineer features in batch
    features_batch = engineer_features_batch(
        valid_sequences, valid_loc_indices, valid_avg_traffic
    )

    # Prepare batch inputs
    model_inputs = prepare_batch_inputs(features_batch, feature_cols)

    # Make batch prediction
    predictions = model.predict(model_inputs, verbose=0, batch_size=args.batch_size)

    # Format results
    results = []
    for loc, pred in zip(valid_locations, predictions):
        results.append(
            {
                "Location": loc,
                "Date": target_date,
                "interval_id": target_interval,
                "time_of_day": f"{target_interval//4:02d}:{(target_interval%4)*15:02d}",
                "predicted_traffic": pred[0],
            }
        )

    return results


# Main prediction process
print("Starting batch rolling predictions...")

# Initialize sequence manager
locations = sorted(df_clean["Location"].unique())
seq_manager = SequenceManager(locations, args.seq_length)
seq_manager.initialize_sequences(df_clean)

all_predictions = []

# Phase 1: Fill October gaps (if any)
print("Phase 1: Checking for October gaps...")
october_gaps = identify_october_gaps(df_clean)

if len(october_gaps) > 0:
    print(f"Found {len(october_gaps)} missing intervals in October")

    start_time = time.time()
    # Group by date and interval
    for (date, interval), group in october_gaps.groupby(["Date", "interval_id"]):
        # Predict for all locations in this interval
        interval_predictions = batch_predict_interval(
            seq_manager,
            group["Location"].tolist(),
            date,
            interval,
            model,
            scaler,
            feature_cols,
        )

        # Update sequences and store predictions
        for pred in interval_predictions:
            seq_manager.update_sequence(
                pred["Location"],
                pred["Date"],
                pred["interval_id"],
                pred["predicted_traffic"],
            )
            all_predictions.append(pred)

    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
else:
    print("No October gaps found")

# Phase 2: Predict November
print("\nPhase 2: Predicting November...")
november_start = pd.Timestamp("2006-11-01")
november_end = pd.Timestamp("2006-11-30")
november_dates = pd.date_range(november_start, november_end, freq="D")

start_time = time.time()
# Process each day in November
for date in november_dates:
    # Process each interval
    for interval in range(96):
        # Batch predict for all locations
        interval_predictions = batch_predict_interval(
            seq_manager, locations, date, interval, model, scaler, feature_cols
        )

        # Update sequences and store predictions
        for pred in interval_predictions:
            seq_manager.update_sequence(
                pred["Location"],
                pred["Date"],
                pred["interval_id"],
                pred["predicted_traffic"],
            )
            all_predictions.append(pred)

elapsed_time = time.time() - start_time
print(f"Total processing time: {elapsed_time:.2f} seconds")

# Convert predictions to DataFrame
df_predictions = pd.DataFrame(all_predictions)

# Save predictions
output_file = os.path.join(MODEL_OUTPUT_PATH, "traffic_predictions_oct_nov_2006.csv")
df_predictions.to_csv(output_file, index=False)
print(f"\nSaved predictions to: {output_file}")

# Also save as pickle for faster loading
pickle_file = os.path.join(MODEL_OUTPUT_PATH, "traffic_predictions_oct_nov_2006.pkl")
with open(pickle_file, "wb") as f:
    pickle.dump(df_predictions, f)

# Create summary
summary = {
    "model_name": args.model_name,
    "total_predictions": len(df_predictions),
    "locations": (
        list(df_predictions["Location"].unique()) if len(df_predictions) > 0 else []
    ),
    "date_range": (
        f"{df_predictions['Date'].min()} to {df_predictions['Date'].max()}"
        if len(df_predictions) > 0
        else "No predictions"
    ),
    "prediction_stats": (
        {
            "mean": (
                df_predictions["predicted_traffic"].mean()
                if len(df_predictions) > 0
                else 0
            ),
            "median": (
                df_predictions["predicted_traffic"].median()
                if len(df_predictions) > 0
                else 0
            ),
            "std": (
                df_predictions["predicted_traffic"].std()
                if len(df_predictions) > 0
                else 0
            ),
            "min": (
                df_predictions["predicted_traffic"].min()
                if len(df_predictions) > 0
                else 0
            ),
            "max": (
                df_predictions["predicted_traffic"].max()
                if len(df_predictions) > 0
                else 0
            ),
        }
        if len(df_predictions) > 0
        else {}
    ),
}

# Save summary
summary_file = os.path.join(MODEL_OUTPUT_PATH, "prediction_summary.pkl")
with open(summary_file, "wb") as f:
    pickle.dump(summary, f)

print(f"\nPrediction Summary:")
print(f"Model: {args.model_name}")
print(f"Total predictions: {summary['total_predictions']}")
print(f"Locations processed: {len(summary['locations'])}")
if len(df_predictions) > 0:
    print(f"Date range: {summary['date_range']}")
    print(f"Mean predicted traffic: {summary['prediction_stats']['mean']:.2f}")
    print(f"Std predicted traffic: {summary['prediction_stats']['std']:.2f}")

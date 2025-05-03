import pandas as pd
import numpy as np
import os
import pickle


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

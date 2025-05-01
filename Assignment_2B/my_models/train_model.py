"""
SCATS Traffic Prediction - Model Training Module
This module provides functions to train, evaluate, and save deep learning models.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model_architecture import create_model


def train_model(
    model,
    X_train_inputs,
    y_train,
    X_val_inputs,
    y_val,
    epochs=50,
    batch_size=64,
    learning_rate=0.001,
    model_dir="checkpoints",
    model_name="traffic_model",
):
    """
    Train a model with early stopping, checkpoints, and learning rate schedules
    """
    print(f"Training model {model_name}...")

    # Format current datetime
    current_time = time.strftime("%Y%m%d_%H%M%S")

    # Create model and log directories if they don't exist
    model_save_dir = os.path.join(model_dir, "saved_models", model_name)
    log_dir = os.path.join(model_dir, "logs", model_name)

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Compile model with Adam optimizer and MSE loss
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"]
    )

    # Prepare callbacks
    callbacks = []

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, verbose=1, restore_best_weights=True
    )
    callbacks.append(early_stopping)

    # Model checkpoint - update file extension from .h5 to .keras
    model_path = os.path.join(model_save_dir, "best.keras")
    checkpoint = ModelCheckpoint(
        filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True
    )
    callbacks.append(checkpoint)

    # Learning rate reducer
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-6
    )
    callbacks.append(reduce_lr)

    # TensorBoard
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
    callbacks.append(tensorboard)
    print(f"TensorBoard logs saved to {log_dir}")

    # Train model
    start_time = time.time()
    history = model.fit(
        X_train_inputs,
        y_train,
        validation_data=(X_val_inputs, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save final model - update file extension from .h5 to .keras
    final_model_path = os.path.join(model_save_dir, "final.keras")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Plot training history
    plot_training_history(history, model_name, model_dir)

    return {
        "model": model,
        "history": history.history,
        "training_time": training_time,
        "model_path": model_save_dir,
    }


def plot_training_history(history, model_name, output_dir):
    """
    Plot training history (loss and metrics)

    Args:
        history: Training history object
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    # Create figure directory if it doesn't exist
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} - Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, f"{model_name}_loss.png"), dpi=300)
    plt.close()

    # Plot MAE
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title(f"{model_name} - Training and Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, f"{model_name}_mae.png"), dpi=300)
    plt.close()


def evaluate_model(
    model, X_test_inputs, y_test, meta_test, model_name, output_dir="evaluation"
):
    """
    Evaluate a trained model on test data and generate evaluation metrics and plots

    Args:
        model: Trained Keras model
        X_test_inputs: Test inputs (list of arrays for feature and location inputs)
        y_test: Test targets
        meta_test: Test metadata
        model_name: Model name for saving results
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating model {model_name} on test data...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate predictions
    y_pred = model.predict(X_test_inputs)
    y_pred = y_pred.flatten()  # Ensure 1D array

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print metrics
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Create results DataFrame
    results = {
        "actual": y_test,
        "predicted": y_pred,
        "error": y_test - y_pred,
        "location": meta_test["Location"],
        "target_date": meta_test["target_date"],
        "target_time": meta_test["target_time"],
    }

    # Create plots
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Plot actual vs predicted
    plot_actual_vs_predicted(y_test, y_pred, model_name, fig_dir)

    # Plot prediction examples for a few locations
    plot_prediction_examples(results, model_name, fig_dir)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "predictions": y_pred}


def plot_actual_vs_predicted(y_true, y_pred, model_name, output_dir):
    """
    Create scatter plot of actual vs predicted values

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Add perfect prediction line
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([0, max_val], [0, max_val], "r--")

    # Add labels and title
    plt.xlabel("Actual Traffic Volume")
    plt.ylabel("Predicted Traffic Volume")
    plt.title(f"{model_name} - Actual vs Predicted Traffic Volume")

    # Add metrics text
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics_text = (
        f"MSE: {mse:.2f}\n" f"RMSE: {rmse:.2f}\n" f"MAE: {mae:.2f}\n" f"R²: {r2:.2f}"
    )
    plt.annotate(
        metrics_text,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_scatter.png"), dpi=300)
    plt.close()

    # Also plot histogram of errors
    plt.figure(figsize=(10, 6))
    plt.hist(y_true - y_pred, bins=50, alpha=0.75)
    plt.axvline(x=0, color="r", linestyle="--")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"{model_name} - Prediction Error Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_error_hist.png"), dpi=300)
    plt.close()


def plot_prediction_examples(results, model_name, output_dir):
    """
    Plot examples of predictions for specific locations

    Args:
        results: Dictionary with actual and predicted values
        model_name: Name of the model
        output_dir: Directory to save the plots
    """
    # Get unique locations
    locations = np.unique(results["location"])

    # Select up to 5 locations
    sample_locations = locations[: min(5, len(locations))]

    for location in sample_locations:
        # Filter data for this location
        mask = results["location"] == location
        dates = results["target_date"][mask]
        times = results["target_time"][mask]
        actual = results["actual"][mask]
        predicted = results["predicted"][mask]

        # Create a timestamp by combining date and time
        timestamps = [f"{date} {time}" for date, time in zip(dates, times)]

        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        timestamps = [timestamps[i] for i in sort_idx]
        actual = actual[sort_idx]
        predicted = predicted[sort_idx]

        # Take only first 100 points to avoid overcrowding
        if len(timestamps) > 100:
            timestamps = timestamps[:100]
            actual = actual[:100]
            predicted = predicted[:100]

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(actual, "b-", label="Actual")
        plt.plot(predicted, "r-", label="Predicted")

        # Add labels and title
        plt.title(f"{model_name} - Predictions for {location}")
        plt.xlabel("Time Point")
        plt.ylabel("Traffic Volume")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Make x-axis labels more readable
        if len(timestamps) > 20:
            plt.xticks(np.arange(0, len(timestamps), len(timestamps) // 10))

        plt.tight_layout()

        # Create safe filename
        safe_location = location.replace(" ", "_").replace("/", "_")
        plt.savefig(
            os.path.join(output_dir, f"{model_name}_{safe_location}_example.png"),
            dpi=300,
        )
        plt.close()


def train_and_evaluate_model(
    data,
    checkpoint_dir="checkpoints",
    model_type="lstm",
    model_params=None,
    train_params=None,
    model_name=None,
):
    """
    Train and evaluate a model with the specified parameters

    Args:
        data: Dictionary with processed data from load_processed_data()
        checkpoint_dir: Directory to save model checkpoints and logs
        model_type: Type of model to create
        model_params: Parameters for model creation
        train_params: Parameters for training
        model_name: Name of the model (will be generated if None)

    Returns:
        Dictionary with training and evaluation results
    """
    # Set default parameters if None
    if model_params is None:
        model_params = {}

    if train_params is None:
        train_params = {}

    # Set model name if None
    if model_name is None:
        # Generate model name based on model_type and current date/time
        current_time = time.strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_{current_time}"

    # Get required parameters from data
    seq_length = data["X_train"].shape[1]
    n_features = data["n_features"]
    n_locations = data["n_locations"]

    # Create model
    model_params.update(
        {"seq_length": seq_length, "n_features": n_features, "n_locations": n_locations}
    )

    print(f"Creating {model_type} model...")
    model = create_model(model_type, **model_params)
    model.summary()

    # Train model
    training_results = train_model(
        model=model,
        model_dir=checkpoint_dir,
        X_train_inputs=data["X_train_inputs"],
        y_train=data["y_train"],
        X_val_inputs=data["X_test_inputs"],
        y_val=data["y_test"],
        model_name=model_name,
        **train_params,
    )

    # Evaluate model
    evaluation_results = evaluate_model(
        model=model,
        X_test_inputs=data["X_test_inputs"],
        y_test=data["y_test"],
        meta_test=data["meta_test"],
        model_name=model_name,
        output_dir=os.path.join(checkpoint_dir, "evaluations", model_name),
    )

    return {
        "model": model,
        "training_results": training_results,
        "evaluation_results": evaluation_results,
    }


if __name__ == "__main__":
    # Load processed data
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_processing.feature_engineering import load_processed_data

    data = load_processed_data()

    # Define models to train
    models_to_train = [
        # {
        #     'type': 'lstm',
        #     'name': 'lstm_basic',
        #     'params': {
        #         'lstm_units': 64,
        #         'dropout_rate': 0.2
        #     }
        # },
        {
            "type": "gru",
            "params": {"gru_units": 64, "dropout_rate": 0.2},
        },
        # {
        #     'type': 'bilstm',
        #     'name': 'bilstm_basic',
        #     'params': {
        #         'lstm_units': 64,
        #         'dropout_rate': 0.2
        #     }
        # }
    ]

    # Train each model
    results = {}
    for model_config in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_config['name']}...")
        print(f"{'='*50}")

        results[model_config["name"]] = train_and_evaluate_model(
            data=data,
            model_type=model_config["type"],
            model_params=model_config["params"],
            model_name=model_config["name"],
        )

    # Print summary of results
    print("\n\nTraining and Evaluation Summary:")
    print(f"{'='*50}")
    for name, result in results.items():
        eval_result = result["evaluation_results"]
        print(f"{name}:")
        print(f"  RMSE: {eval_result['rmse']:.4f}")
        print(f"  MAE: {eval_result['mae']:.4f}")
        print(f"  R²: {eval_result['r2']:.4f}")

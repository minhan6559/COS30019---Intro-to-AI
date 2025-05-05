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
from src.train_and_evaluate.model_architecture import create_model


def train_model(
    model,
    X_train_inputs,
    y_train,
    X_val_inputs,
    y_val,
    epochs=50,
    batch_size=128,
    learning_rate=0.001,
    clipnorm=1.0,
    early_stopping_patience=10,
    reduce_lr_patience=3,
    verbose=1,
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
        optimizer=Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss="mse",
        metrics=["mae"],
    )

    # Prepare callbacks
    callbacks = []

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        verbose=verbose,
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)

    # Model checkpoint - update file extension from .h5 to .keras
    model_path = os.path.join(model_save_dir, "best.keras")
    checkpoint = ModelCheckpoint(
        filepath=model_path, monitor="val_loss", verbose=verbose, save_best_only=True
    )
    callbacks.append(checkpoint)

    # Learning rate reducer
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=reduce_lr_patience,  # Reduced from 5 to react faster
        verbose=1,
        min_lr=1e-6,
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
        verbose=verbose,
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save final model - update file extension from .h5 to .keras
    final_model_path = os.path.join(model_save_dir, f"final_{model_name}.keras")
    model.save(final_model_path)

    # Save final model architecture as JSON
    model_json = model.to_json()
    final_model_json_path = os.path.join(model_save_dir, f"final_{model_name}.json")
    with open(final_model_json_path, "w") as json_file:
        json_file.write(model_json)

    # Save weights
    weights = model.get_weights()
    weights_path = os.path.join(model_save_dir, f"final_{model_name}_weights.npz")
    np.savez(weights_path, *weights)

    print(f"Model saved to {model_save_dir} folder")

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
    fig_dir = os.path.join(output_dir, "figures", model_name)
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
    model,
    X_test_inputs,
    y_test,
    meta_test,
    model_name,
    output_dir="evaluations",
    max_visualization_points=288,
    num_visualization_locations=10,
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
    plot_prediction_examples(
        results,
        model_name,
        fig_dir,
        max_points=max_visualization_points,
        num_locations=num_visualization_locations,
    )

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


def plot_prediction_examples(
    results, model_name, output_dir, max_points=288, num_locations=10
):
    """
    Plot examples of predictions for specific locations with time values on x-axis

    Args:
        results: Dictionary with actual and predicted values
        model_name: Name of the model
        output_dir: Directory to save the plots
        max_points: Maximum number of timepoints to visualize (default: 288)
        num_locations: Number of locations to sample (default: 10)
    """
    # Get unique locations
    locations = np.unique(results["location"])

    # Select up to num_locations locations
    sample_locations = locations[: min(num_locations, len(locations))]

    for location in sample_locations:
        # Filter data for this location
        mask = results["location"] == location

        # Convert to arrays to avoid indexing issues
        dates = np.array(results["target_date"][mask])
        times = np.array(results["target_time"][mask])
        actual = np.array(results["actual"][mask])
        predicted = np.array(results["predicted"][mask])

        # Create a timestamp by combining date and time
        timestamps = [f"{d} {t}" for d, t in zip(dates, times)]

        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        timestamps = [timestamps[i] for i in sort_idx]
        actual = actual[sort_idx]
        predicted = predicted[sort_idx]
        sorted_times = [times[i] for i in sort_idx]  # Now this should work

        # Take only max_points to avoid overcrowding
        if len(timestamps) > max_points:
            timestamps = timestamps[:max_points]
            actual = actual[:max_points]
            predicted = predicted[:max_points]
            sorted_times = sorted_times[:max_points]

        # Plot
        plt.figure(figsize=(14, 6))
        plt.plot(actual, "b-", label="Actual", linewidth=1.5)
        plt.plot(predicted, "r-", label="Predicted", linewidth=1.5)

        # Create x-axis labels with time values
        if len(sorted_times) > 200:
            tick_spacing = len(sorted_times) // 20  # Show ~20 ticks
        elif len(sorted_times) > 100:
            tick_spacing = len(sorted_times) // 15  # Show ~15 ticks
        else:
            tick_spacing = max(1, len(sorted_times) // 10)  # Show ~10 ticks

        # Set x-ticks at regular intervals
        tick_positions = range(0, len(sorted_times), tick_spacing)
        tick_labels = [sorted_times[i] for i in tick_positions]

        plt.xticks(tick_positions, tick_labels, rotation=45)

        # Add labels and title
        plt.title(
            f"{model_name} - Predictions for {location}\n(showing {len(timestamps)} points)"
        )
        plt.xlabel("Time of Day")
        plt.ylabel("Traffic Volume")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Create safe filename
        safe_location = location.replace(" ", "_").replace("/", "_")
        plt.savefig(
            os.path.join(
                output_dir,
                f"{model_name}_{safe_location}_{max_points}points_example.png",
            ),
            dpi=300,
        )
        plt.close()


if __name__ == "__main__":
    # Load processed data
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_preprocessing.feature_engineering import load_processed_data

    from model_architecture import create_gru_model

    data = load_processed_data()
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define models to train
    models_to_train = [
        {
            "type": "gru",
            "create_model_func": create_gru_model,
            "model_params": {"gru_units": 64, "dropout_rate": 0.2},
            "train_params": {"epochs": 50, "batch_size": 64},
        },
    ]

    # Train each model
    results = {}
    for model_config in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_config['name']}...")
        print(f"{'='*50}")

        # Set default parameters if None
        model_params = model_config.get("model_params", {})
        train_params = model_config.get("train_params", {})
        model_type = model_config["type"]

        current_time = time.strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_{current_time}"

        # Get required parameters from data
        seq_length = data["X_train"].shape[1]
        n_features = data["n_features"]
        n_locations = data["n_locations"]

        # Create model
        model_params.update(
            {
                "seq_length": seq_length,
                "n_features": n_features,
                "n_locations": n_locations,
            }
        )

        print(f"Creating {model_type} model...")

        if (
            "create_model_func" in model_config
        ):  # Check if a custom model creation function is provided
            create_model_func = model_config["create_model_func"]
            model = create_model_func(**model_params)
        else:  # Use default create_model function
            model = create_model(model_type=model_type, **model_params)

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

        results[model_config["name"]] = {
            "model": model,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
        }

    # Print summary of results
    print("\n\nTraining and Evaluation Summary:")
    print(f"{'='*50}")
    for name, result in results.items():
        eval_result = result["evaluation_results"]
        print(f"{name}:")
        print(f"  RMSE: {eval_result['rmse']:.4f}")
        print(f"  MAE: {eval_result['mae']:.4f}")
        print(f"  R²: {eval_result['r2']:.4f}")

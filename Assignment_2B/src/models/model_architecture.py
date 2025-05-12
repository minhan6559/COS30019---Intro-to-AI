"""
SCATS Traffic Prediction - Model Architecture Module
This module provides functions to create various deep learning model architectures.
"""

import tensorflow as tf
from tensorflow import keras

# Import Keras modules directly
# Core TensorFlow and Keras imports
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    # Input and basic layers
    Input,
    Dense,
    Dropout,
    Activation,
    Reshape,
    RepeatVector,
    Lambda,
    Concatenate,
    Add,
    Subtract,
    Multiply,
    Average,
    # Normalization and regularization
    BatchNormalization,
    LayerNormalization,
    SpatialDropout1D,
    # Embedding layers
    Embedding,
    # Convolutional layers
    Conv1D,
    MaxPooling1D,
    AveragePooling1D,
    # Recurrent layers
    LSTM,
    GRU,
    Bidirectional,
    # Attention mechanisms
    MultiHeadAttention,
    Attention,
    # Pooling layers
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    # Shape operations
    Flatten,
    Dot,
    Softmax,
)
from keras.regularizers import l2


def create_lstm_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=12,
    lstm_units_list=[48, 24],  # Now a list of units for multiple LSTM layers
    dense_units_list=[32, 16],
    dropout_rate=0.3,
    l2_reg=0.001,
    recurrent_l2=0.001,
    activation="relu",
):
    """
    Create an enhanced stacked LSTM model with configurable recurrent and dense layers

    Args:
        seq_length: Length of input sequences
        n_features: Number of features
        n_locations: Number of unique locations
        embedding_dim: Dimension of location embedding
        lstm_units_list: List of units for each LSTM layer
        dense_units_list: List of units for each dense layer
        dropout_rate: Dropout rate
        l2_reg: L2 regularization factor
        recurrent_l2: L2 regularization factor for recurrent weights
        activation: Activation function for dense layers

    Returns:
        Keras model
    """
    # Convert the LSTM units to a list if it's a single integer
    if isinstance(lstm_units_list, int):
        lstm_units_list = [lstm_units_list]

    # Convert the dense units to a list if it's a single integer
    if isinstance(dense_units_list, int):
        dense_units_list = [dense_units_list]

    # Input layers
    feature_input = Input(shape=(seq_length, n_features), name="feature_input")
    location_input = Input(shape=(seq_length,), dtype="int32", name="location_input")

    # Location embedding with regularization
    location_embedding = Embedding(
        input_dim=n_locations,
        output_dim=embedding_dim,
        embeddings_regularizer=l2(l2_reg),
        name="location_embedding",
    )(location_input)

    # Combine feature input with location embedding
    combined_input = Concatenate(axis=2)([feature_input, location_embedding])

    # Add BatchNormalization before LSTM layers
    x = BatchNormalization(name="input_normalization")(combined_input)

    # Create stacked LSTM layers
    for i, units in enumerate(lstm_units_list):
        # Determine whether to return sequences (all but last layer should return sequences)
        return_sequences = i < len(lstm_units_list) - 1

        # LSTM layer with regularization
        x = LSTM(
            units=units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(recurrent_l2),
            activity_regularizer=l2(l2_reg / 10),
            name=f"lstm_layer_{i+1}",
        )(x)

        # Add batch normalization between LSTM layers
        x = BatchNormalization(name=f"lstm_norm_{i+1}")(x)

        # Add dropout between LSTM layers
        if return_sequences:
            x = Dropout(dropout_rate, name=f"lstm_dropout_{i+1}")(x)

    # Apply final dropout after the last LSTM layer
    x = Dropout(dropout_rate, name="final_lstm_dropout")(x)

    # Dynamically create dense layers based on the dense_units_list
    for i, units in enumerate(dense_units_list):
        x = Dense(
            units,
            activation=activation,
            kernel_regularizer=l2(l2_reg),
            name=f"dense_layer_{i+1}",
        )(x)
        x = BatchNormalization(name=f"dense_norm_{i+1}")(x)
        x = Dropout(dropout_rate, name=f"dense_dropout_{i+1}")(x)

    # Output layer
    output = Dense(1, activation="linear", name="output")(x)

    # Create model
    model = Model(inputs=[feature_input, location_input], outputs=output)

    return model


def create_bidirectional_lstm_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=12,
    lstm_units=48,
    dense_units_list=[32, 16],  # List of units for dense layers
    dropout_rate=0.3,
    l2_reg=0.001,
    recurrent_l2=0.001,
    activation="relu",
    use_attention=False,
):
    """
    Create an enhanced Bidirectional LSTM model with configurable dense layers and regularization

    Args:
        seq_length: Length of input sequences
        n_features: Number of features
        n_locations: Number of unique locations
        embedding_dim: Dimension of location embedding
        lstm_units: Number of units in LSTM layer
        dense_units_list: List of units for each dense layer
        dropout_rate: Dropout rate
        l2_reg: L2 regularization factor
        recurrent_l2: L2 regularization factor for recurrent weights
        activation: Activation function for dense layers
        use_attention: Whether to use attention mechanism

    Returns:
        Keras model
    """
    # Input layers
    feature_input = Input(shape=(seq_length, n_features), name="feature_input")
    location_input = Input(shape=(seq_length,), dtype="int32", name="location_input")

    # Location embedding with regularization
    location_embedding = Embedding(
        input_dim=n_locations,
        output_dim=embedding_dim,
        embeddings_regularizer=l2(l2_reg),
        name="location_embedding",
    )(location_input)

    # Combine feature input with location embedding
    combined_input = Concatenate(axis=2)([feature_input, location_embedding])

    # Add BatchNormalization before BiLSTM
    normalized_input = BatchNormalization(name="input_normalization")(combined_input)

    # Bidirectional LSTM layer with regularization
    bilstm_layer = Bidirectional(
        LSTM(
            units=lstm_units,
            return_sequences=use_attention,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(recurrent_l2),
            activity_regularizer=l2(l2_reg / 10),
            name="lstm_layer",
        ),
        name="bidirectional_wrapper",
    )(normalized_input)

    # Add attention if requested
    if use_attention:
        # Self-attention mechanism
        attention_layer = tf.keras.layers.Attention()([bilstm_layer, bilstm_layer])
        # Global pooling to get a fixed-size output
        x = GlobalAveragePooling1D()(attention_layer)
    else:
        x = bilstm_layer

    # Apply dropout and batch normalization to BiLSTM output
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization(name="bilstm_output_norm")(x)

    # Dynamically create dense layers based on the dense_units_list
    for i, units in enumerate(dense_units_list):
        x = Dense(
            units,
            activation=activation,
            kernel_regularizer=l2(l2_reg),
            name=f"dense_layer_{i+1}",
        )(x)
        x = BatchNormalization(name=f"dense_norm_{i+1}")(x)
        x = Dropout(dropout_rate)(x)

    # Output layer
    output = Dense(1, activation="linear", name="output")(x)

    # Create model
    model = Model(inputs=[feature_input, location_input], outputs=output)

    return model


def create_gru_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=12,
    gru_units_list=[48],
    dense_units_list=[32, 16],
    dropout_rate=0.3,
    l2_reg=0.001,
    recurrent_l2=0.001,
    activation="relu",
):
    """
    Create an enhanced stacked GRU model with configurable recurrent and dense layers

    Args:
        seq_length: Length of input sequences
        n_features: Number of features
        n_locations: Number of unique locations
        embedding_dim: Dimension of location embedding
        gru_units_list: List of units for each GRU layer
        dense_units_list: List of units for each dense layer
        dropout_rate: Dropout rate
        l2_reg: L2 regularization factor
        recurrent_l2: L2 regularization factor for recurrent weights
        activation: Activation function for dense layers

    Returns:
        Keras model
    """
    # Convert the GRU units to a list if it's a single integer
    if isinstance(gru_units_list, int):
        gru_units_list = [gru_units_list]

    # Convert the dense units to a list if it's a single integer
    if isinstance(dense_units_list, int):
        dense_units_list = [dense_units_list]

    # Input layers
    feature_input = Input(shape=(seq_length, n_features), name="feature_input")
    location_input = Input(shape=(seq_length,), dtype="int32", name="location_input")

    # Location embedding with regularization
    location_embedding = Embedding(
        input_dim=n_locations,
        output_dim=embedding_dim,
        embeddings_regularizer=l2(l2_reg),
        name="location_embedding",
    )(location_input)

    # Combine feature input with location embedding
    combined_input = Concatenate(axis=2)([feature_input, location_embedding])

    # Add BatchNormalization before GRU layers
    x = BatchNormalization(name="input_normalization")(combined_input)

    # Create stacked GRU layers
    for i, units in enumerate(gru_units_list):
        # Determine whether to return sequences (all but last layer should return sequences)
        return_sequences = i < len(gru_units_list) - 1

        # GRU layer with regularization
        x = GRU(
            units=units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(recurrent_l2),
            activity_regularizer=l2(l2_reg / 10),
            name=f"gru_layer_{i+1}",
        )(x)

        # Add batch normalization between GRU layers
        x = BatchNormalization(name=f"gru_norm_{i+1}")(x)

        # Add dropout between GRU layers
        if return_sequences:
            x = Dropout(dropout_rate, name=f"gru_dropout_{i+1}")(x)

    # Apply final dropout after the last GRU layer
    x = Dropout(dropout_rate, name="final_gru_dropout")(x)

    # Dynamically create dense layers based on the dense_units_list
    for i, units in enumerate(dense_units_list):
        x = Dense(
            units,
            activation=activation,
            kernel_regularizer=l2(l2_reg),
            name=f"dense_layer_{i+1}",
        )(x)
        x = BatchNormalization(name=f"dense_norm_{i+1}")(x)
        x = Dropout(dropout_rate, name=f"dense_dropout_{i+1}")(x)

    # Output layer
    output = Dense(1, activation="linear", name="output")(x)

    # Create model
    model = Model(inputs=[feature_input, location_input], outputs=output)

    return model


def create_bidirectional_gru_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=16,
    gru_units=64,
    dropout_rate=0.2,
    use_attention=False,
):
    """
    Create a model with Bidirectional GRU architecture and location embedding

    Args:
        seq_length: Length of input sequences
        n_features: Number of features (excluding location_idx)
        n_locations: Number of unique locations
        embedding_dim: Dimension of location embedding
        gru_units: Number of units in GRU layer
        dropout_rate: Dropout rate
        use_attention: Whether to use attention mechanism

    Returns:
        Keras model
    """
    # Input layers
    feature_input = Input(shape=(seq_length, n_features), name="feature_input")
    location_input = Input(shape=(seq_length,), dtype="int32", name="location_input")

    # Location embedding
    location_embedding = Embedding(
        input_dim=n_locations, output_dim=embedding_dim, name="location_embedding"
    )(location_input)

    # Combine feature input with location embedding
    combined_input = Concatenate(axis=2)([feature_input, location_embedding])

    # Bidirectional GRU layer
    bigru_layer = Bidirectional(
        GRU(
            units=gru_units,
            return_sequences=use_attention,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name="gru_layer",
        ),
        name="bidirectional_wrapper",
    )(combined_input)

    # Add attention if requested
    if use_attention:
        # Simple self-attention
        attention_layer = tf.keras.layers.Attention()([bigru_layer, bigru_layer])
        # Global pooling to get a fixed-size output
        bigru_output = GlobalAveragePooling1D()(attention_layer)
    else:
        bigru_output = bigru_layer

    # Batch normalization
    normalized = BatchNormalization(name="batch_norm")(bigru_output)

    # Dense layers
    dense1 = Dense(32, activation="relu", name="dense1")(normalized)
    dense1 = Dropout(dropout_rate)(dense1)

    # Output layer
    output = Dense(1, activation="linear", name="output")(dense1)

    # Create model
    model = Model(inputs=[feature_input, location_input], outputs=output)

    return model


def create_cnn_lstm_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=16,
    filters=64,
    kernel_size=3,
    lstm_units=64,
    dropout_rate=0.2,
):
    """
    Create a model with CNN + LSTM architecture and location embedding

    Args:
        seq_length: Length of input sequences
        n_features: Number of features (excluding location_idx)
        n_locations: Number of unique locations
        embedding_dim: Dimension of location embedding
        filters: Number of CNN filters
        kernel_size: CNN kernel size
        lstm_units: Number of units in LSTM layer
        dropout_rate: Dropout rate

    Returns:
        Keras model
    """
    # Input layers
    feature_input = Input(shape=(seq_length, n_features), name="feature_input")
    location_input = Input(shape=(seq_length,), dtype="int32", name="location_input")

    # Location embedding
    location_embedding = Embedding(
        input_dim=n_locations, output_dim=embedding_dim, name="location_embedding"
    )(location_input)

    # Combine feature input with location embedding
    combined_input = Concatenate(axis=2)([feature_input, location_embedding])

    # CNN layers for feature extraction
    conv1 = Conv1D(
        filters=filters, kernel_size=kernel_size, activation="relu", padding="same"
    )(combined_input)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(
        filters=filters * 2, kernel_size=kernel_size, activation="relu", padding="same"
    )(pool1)

    # LSTM layer for sequential modeling
    lstm_layer = LSTM(units=lstm_units)(conv2)
    lstm_layer = Dropout(dropout_rate)(lstm_layer)

    # Output layer
    output = Dense(1, activation="linear", name="output")(lstm_layer)

    # Create model
    model = Model(inputs=[feature_input, location_input], outputs=output)

    return model


def create_transformer_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=16,
    num_heads=4,
    ff_dim=64,
    num_transformer_blocks=2,
    dropout_rate=0.2,
):
    """
    Create a Transformer model with location embedding

    Args:
        seq_length: Length of input sequences
        n_features: Number of features (excluding location_idx)
        n_locations: Number of unique locations
        embedding_dim: Dimension of location embedding
        num_heads: Number of attention heads
        ff_dim: Hidden layer size in feed forward network inside transformer
        num_transformer_blocks: Number of transformer blocks
        dropout_rate: Dropout rate

    Returns:
        Keras model
    """
    # Input layers
    feature_input = Input(shape=(seq_length, n_features), name="feature_input")
    location_input = Input(shape=(seq_length,), dtype="int32", name="location_input")

    # Location embedding
    location_embedding = Embedding(
        input_dim=n_locations, output_dim=embedding_dim, name="location_embedding"
    )(location_input)

    # Combine feature input with location embedding
    combined_input = Concatenate(axis=2)([feature_input, location_embedding])

    # Transformer blocks
    x = combined_input
    for i in range(num_transformer_blocks):
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=n_features + embedding_dim // num_heads
        )(x, x)
        attention_output = Dropout(dropout_rate)(attention_output)
        # Add & Norm (first residual connection)
        x1 = Add()([x, attention_output])
        x1 = LayerNormalization(epsilon=1e-6)(x1)

        # Feed-forward network
        ffn_output = Dense(ff_dim, activation="relu")(x1)
        ffn_output = Dense(n_features + embedding_dim)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        # Add & Norm (second residual connection)
        x = Add()([x1, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

    # Global pooling to get a fixed-size representation
    x = GlobalAveragePooling1D()(x)

    # Dense output layers
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation="linear", name="output")(x)

    # Create model
    model = Model(inputs=[feature_input, location_input], outputs=output)

    return model


# Dictionary mapping model names to their creation functions
MODEL_REGISTRY = {
    "lstm": create_lstm_model,
    "bilstm": create_bidirectional_lstm_model,
    "gru": create_gru_model,
    "bigru": create_bidirectional_gru_model,
    "cnn_lstm": create_cnn_lstm_model,
    "transformer": create_transformer_model,
}


def create_model(model_type, **kwargs):
    """
    Create a model of the specified type with given parameters

    Args:
        model_type: Type of model to create (must be in MODEL_REGISTRY)
        **kwargs: Arguments to pass to the model creation function

    Returns:
        Keras model
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Available types: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_type](**kwargs)

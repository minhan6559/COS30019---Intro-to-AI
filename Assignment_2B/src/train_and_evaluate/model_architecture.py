"""
SCATS Traffic Prediction - Model Architecture Module
This module provides functions to create various deep learning model architectures.
"""

import tensorflow as tf

# Import Keras modules directly
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    LSTM,
    GRU,
    Bidirectional,
    Dropout,
    Embedding,
    Concatenate,
    BatchNormalization,
    Conv1D,
    MaxPooling1D,
    Attention,
    GlobalAveragePooling1D,
    LayerNormalization,
    MultiHeadAttention,
    Add,
)
from keras.regularizers import l2


def create_lstm_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=16,
    lstm_units=64,
    dropout_rate=0.2,
    use_attention=False,
):
    """
    Create a model with LSTM architecture and location embedding

    Args:
        seq_length: Length of input sequences
        n_features: Number of features (excluding location_idx)
        n_locations: Number of unique locations
        embedding_dim: Dimension of location embedding
        lstm_units: Number of units in LSTM layer
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

    # LSTM layer
    lstm_layer = LSTM(
        units=lstm_units,
        return_sequences=use_attention,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name="lstm_layer",
    )(combined_input)

    # Add attention if requested
    if use_attention:
        # Simple self-attention
        attention_layer = tf.keras.layers.Attention()([lstm_layer, lstm_layer])
        # Global pooling to get a fixed-size output
        lstm_output = GlobalAveragePooling1D()(attention_layer)
    else:
        lstm_output = lstm_layer

    # Output layer
    output = Dense(1, activation="linear", name="output")(lstm_output)

    # Create model
    model = Model(inputs=[feature_input, location_input], outputs=output)

    return model


def create_bidirectional_lstm_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=16,
    lstm_units=64,
    dropout_rate=0.2,
    use_attention=False,
):
    """
    Create a model with Bidirectional LSTM architecture and location embedding

    Args:
        seq_length: Length of input sequences
        n_features: Number of features (excluding location_idx)
        n_locations: Number of unique locations
        embedding_dim: Dimension of location embedding
        lstm_units: Number of units in LSTM layer
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

    # Bidirectional LSTM layer
    bilstm_layer = Bidirectional(
        LSTM(
            units=lstm_units,
            return_sequences=use_attention,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name="lstm_layer",
        ),
        name="bidirectional_wrapper",
    )(combined_input)

    # Add attention if requested
    if use_attention:
        # Simple self-attention
        attention_layer = tf.keras.layers.Attention()([bilstm_layer, bilstm_layer])
        # Global pooling to get a fixed-size output
        bilstm_output = GlobalAveragePooling1D()(attention_layer)
    else:
        bilstm_output = bilstm_layer

    # Batch normalization
    normalized = BatchNormalization(name="batch_norm")(bilstm_output)

    # Dense layers
    dense1 = Dense(32, activation="relu", name="dense1")(normalized)
    dense1 = Dropout(dropout_rate)(dense1)

    # Output layer
    output = Dense(1, activation="linear", name="output")(dense1)

    # Create model
    model = Model(inputs=[feature_input, location_input], outputs=output)

    return model


def create_gru_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=16,
    gru_units=64,
    dropout_rate=0.2,
    use_attention=False,
):
    """
    Create a model with GRU architecture and location embedding

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

    # GRU layer
    gru_layer = GRU(
        units=gru_units,
        return_sequences=use_attention,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name="gru_layer",
    )(combined_input)

    # Add attention if requested
    if use_attention:
        # Simple self-attention
        attention_layer = tf.keras.layers.Attention()([gru_layer, gru_layer])
        # Global pooling to get a fixed-size output
        gru_output = GlobalAveragePooling1D()(attention_layer)
    else:
        gru_output = gru_layer

    # Output layer
    output = Dense(1, activation="linear", name="output")(gru_output)

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


def create_stacked_lstm_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=16,
    lstm_units_list=[64, 32],
    dropout_rate=0.2,
):
    """
    Create a stacked LSTM model with location embedding

    Args:
        seq_length: Length of input sequences
        n_features: Number of features (excluding location_idx)
        n_locations: Number of unique locations
        embedding_dim: Dimension of location embedding
        lstm_units_list: List of units for each LSTM layer
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

    # Stacked LSTM layers
    x = combined_input
    for i, units in enumerate(lstm_units_list):
        return_sequences = i < len(lstm_units_list) - 1
        x = LSTM(
            units=units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name=f"lstm_layer_{i+1}",
        )(x)

        # Add batch normalization between LSTM layers
        if return_sequences:
            x = BatchNormalization()(x)

    # Fully connected layers
    x = BatchNormalization()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout_rate)(x)

    # Output layer
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
    "stacked_lstm": create_stacked_lstm_model,
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

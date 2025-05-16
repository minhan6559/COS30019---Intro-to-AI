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


def create_enhanced_cnn_lstm_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=12,
    cnn_filters_list=[64, 128],  # List of filters for each CNN layer
    kernel_size=3,
    lstm_units_list=[48, 24],  # List of units for each LSTM layer
    dense_units_list=[32, 16],  # List of units for each dense layer
    dropout_rate=0.3,
    l2_reg=0.001,
    recurrent_l2=0.001,
    attention_heads=4,
    attention_size=32,
    attention_position="after_cnn",  # Options: "after_cnn", "after_lstm", "both"
    activation="relu",
    pooling_type="max",  # Options: "max", "avg", "none"
    pooling_size=2,
):
    """
    Create an enhanced CNN-LSTM model with multihead attention and configurable layers

    Args:
        seq_length: Length of input sequences
        n_features: Number of features
        n_locations: Number of unique locations
        embedding_dim: Dimension of location embedding
        cnn_filters_list: List of filters for each CNN layer
        kernel_size: Kernel size for CNN layers
        lstm_units_list: List of units for each LSTM layer
        dense_units_list: List of units for each dense layer
        dropout_rate: Dropout rate
        l2_reg: L2 regularization factor
        recurrent_l2: L2 regularization factor for recurrent weights
        attention_heads: Number of attention heads
        attention_size: Size of attention dimension
        attention_position: Where to apply attention
        activation: Activation function for layers
        pooling_type: Type of pooling after CNN layers
        pooling_size: Size of pooling window

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

    # Add BatchNormalization before CNN layers
    x = BatchNormalization(name="input_normalization")(combined_input)

    # Create stacked CNN layers with progressive filters
    for i, filters in enumerate(cnn_filters_list):
        # CNN layer with regularization
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding="same",
            kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg / 2),
            name=f"conv_layer_{i+1}",
        )(x)

        # Add batch normalization
        x = BatchNormalization(name=f"conv_norm_{i+1}")(x)

        # Add dropout after each CNN layer
        x = Dropout(dropout_rate * 0.5, name=f"conv_dropout_{i+1}")(x)

        # Add pooling if specified
        if pooling_type.lower() == "max" and i < len(cnn_filters_list) - 1:
            x = MaxPooling1D(pool_size=pooling_size, name=f"max_pool_{i+1}")(x)
        elif pooling_type.lower() == "avg" and i < len(cnn_filters_list) - 1:
            x = AveragePooling1D(pool_size=pooling_size, name=f"avg_pool_{i+1}")(x)

    # Apply attention after CNN if specified
    if attention_position.lower() in ["after_cnn", "both"]:
        # Multihead attention with regularization
        attn_output = MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=attention_size // attention_heads,
            dropout=dropout_rate,
            kernel_regularizer=l2(l2_reg),
            name="multihead_attention_cnn",
        )(x, x)

        # Add & normalize
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6, name="attn_norm_cnn")(x)

    # Create stacked LSTM layers
    for i, units in enumerate(lstm_units_list):
        # Determine whether to return sequences
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

        # Add batch normalization
        x = BatchNormalization(name=f"lstm_norm_{i+1}")(x)

        # Apply attention after each LSTM layer if specified
        if return_sequences and attention_position.lower() in ["after_lstm", "both"]:
            # Multihead attention with regularization
            attn_output = MultiHeadAttention(
                num_heads=attention_heads,
                key_dim=attention_size // attention_heads,
                dropout=dropout_rate,
                kernel_regularizer=l2(l2_reg),
                name=f"multihead_attention_lstm_{i+1}",
            )(x, x)

            # Add & normalize
            x = Add()([x, attn_output])
            x = LayerNormalization(epsilon=1e-6, name=f"attn_norm_lstm_{i+1}")(x)

        # Add dropout between LSTM layers
        if return_sequences:
            x = Dropout(dropout_rate, name=f"lstm_dropout_{i+1}")(x)

    # Final dropout after LSTM
    x = Dropout(dropout_rate, name="final_lstm_dropout")(x)

    # Create dense layers
    for i, units in enumerate(dense_units_list):
        x = Dense(
            units,
            activation=activation,
            kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg / 2),
            name=f"dense_layer_{i+1}",
        )(x)
        x = BatchNormalization(name=f"dense_norm_{i+1}")(x)
        x = Dropout(dropout_rate, name=f"dense_dropout_{i+1}")(x)

    # Output layer
    output = Dense(
        1, activation="linear", kernel_regularizer=l2(l2_reg / 2), name="output"
    )(x)

    # Create model
    model = Model(inputs=[feature_input, location_input], outputs=output)

    return model


def get_positional_encoding(seq_length, d_model):
    """
    Generates positional encoding for a given sequence length and model dimension.

    Args:
        seq_length: Length of the sequence.
        d_model: Dimension of the model (embedding dimension).

    Returns:
        A tensor of shape (1, seq_length, d_model) with positional encodings.
    """
    positions = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(
        tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model)
    )
    pos_encoding = tf.zeros((seq_length, d_model))

    # Apply sin to even indices in the array; 2i
    pos_encoding_sin = tf.sin(positions * div_term)
    # Apply cos to odd indices in the array; 2i+1
    pos_encoding_cos = tf.cos(positions * div_term)

    # Interleave sin and cos
    # This is a common way, but Keras's MHA might expect them concatenated or handled differently.
    # For simplicity and common practice, we'll use this interleaving approach.
    # However, a simpler approach is to assign sin to even indices and cos to odd indices directly.
    indices_sin = tf.range(0, d_model, 2)
    indices_cos = tf.range(1, d_model, 2)

    # Ensure pos_encoding_sin and pos_encoding_cos have the correct shape for scatter_nd
    # They should be (seq_length, d_model/2)
    # If d_model is odd, the last column of pos_encoding_cos might be problematic.
    # Let's ensure d_model is even or handle the odd case.
    # For now, assume d_model is usually even for this type of encoding.

    # Create updates for scatter_nd
    updates_sin = tf.reshape(pos_encoding_sin, [-1])
    updates_cos = tf.reshape(pos_encoding_cos, [-1])

    # Create indices for scatter_nd
    idx_sin = []
    for i in range(seq_length):
        for j_idx, j_val in enumerate(indices_sin):
            idx_sin.append([i, j_val.numpy()])

    idx_cos = []
    for i in range(seq_length):
        for j_idx, j_val in enumerate(indices_cos):
            # Ensure j_val is within bounds for pos_encoding_cos columns
            if j_idx < pos_encoding_cos.shape[1]:
                idx_cos.append([i, j_val.numpy()])

    pos_encoding = tf.tensor_scatter_nd_update(pos_encoding, idx_sin, updates_sin)
    if idx_cos:  # only update if there are odd indices
        pos_encoding = tf.tensor_scatter_nd_update(pos_encoding, idx_cos, updates_cos)

    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, seq_length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.pos_encoding = get_positional_encoding(seq_length, embedding_dim)

    def call(self, x):
        return x + self.pos_encoding[:, : tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "seq_length": self.seq_length,
                "embedding_dim": self.embedding_dim,
            }
        )
        return config


def create_transformer_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=16,
    num_heads=4,
    ff_dim=64,
    num_transformer_blocks=2,
    dropout_rate=0.2,
    use_positional_encoding=True,
):
    """
    Create a Transformer model with location embedding and optional positional encoding.

    Args:
        seq_length: Length of input sequences
        n_features: Number of features (excluding location_idx)
        n_locations: Number of unique locations
        embedding_dim: Dimension of location embedding
        num_heads: Number of attention heads
        ff_dim: Hidden layer size in feed forward network inside transformer
        num_transformer_blocks: Number of transformer blocks
        dropout_rate: Dropout rate
        use_positional_encoding: Boolean, whether to add positional encoding.

    Returns:
        Keras model
    """
    # Input layers
    feature_input = Input(shape=(seq_length, n_features), name="feature_input")
    location_input = Input(shape=(seq_length,), dtype="int32", name="location_input")

    # Location embedding
    location_embedding_layer = Embedding(
        input_dim=n_locations, output_dim=embedding_dim, name="location_embedding"
    )
    location_embedded = location_embedding_layer(location_input)

    # Combine feature input with location embedding
    combined_input = Concatenate(axis=2)([feature_input, location_embedded])

    # Effective dimension after concatenation
    d_model = n_features + embedding_dim

    x = combined_input
    if use_positional_encoding:
        x = PositionalEncoding(seq_length, d_model)(x)
        x = Dropout(dropout_rate)(x)  # Dropout after adding positional encoding

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head self-attention
        # key_dim is the dimension of each head. d_model must be divisible by num_heads.
        # If key_dim is not set, it defaults to d_model // num_heads.
        # The original code had `key_dim=n_features + embedding_dim // num_heads` which is likely incorrect.
        # It should be `d_model // num_heads` or simply let it default.
        # Let's ensure d_model is used correctly.
        if d_model % num_heads != 0:
            # Adjust d_model or num_heads, or use a projection if necessary.
            # For simplicity, we'll assume d_model is divisible by num_heads or use default key_dim.
            # If key_dim is explicitly set, it's the size of query/key/value for each head.
            # The total size across heads should match d_model for the output of MHA to be Add()ed.
            # Keras MHA handles this by projecting Q, K, V to key_dim * num_heads and then outputting d_model.
            # So, setting key_dim to d_model // num_heads is standard.
            attention_key_dim = d_model // num_heads
        else:
            attention_key_dim = d_model // num_heads

        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=attention_key_dim  # d_model // num_heads
        )(
            query=x, value=x, key=x
        )  # Using x for query, key, and value for self-attention
        attention_output = Dropout(dropout_rate)(attention_output)
        # Add & Norm (first residual connection)
        x1 = Add()([x, attention_output])
        x1 = LayerNormalization(epsilon=1e-6)(x1)

        # Feed-forward network
        ffn_output = Dense(ff_dim, activation="relu")(x1)
        ffn_output = Dense(d_model)(ffn_output)  # Project back to d_model
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


def create_tcn_model(
    seq_length,
    n_features,
    n_locations,
    embedding_dim=10,
    nb_filters=64,
    kernel_size=3,
    nb_stacks=2,
    dilations=[1, 2, 4, 8, 16, 32],
    dropout_rate=0.3,
    l2_reg=0.001,
    dense_units_list=[32, 16],
    activation="elu",
):
    """
    Create a Temporal Convolutional Network model for traffic prediction

    Args:
        seq_length: Length of input sequences
        n_features: Number of features
        n_locations: Number of unique locations
        embedding_dim: Dimension of location embedding
        nb_filters: Number of filters in convolutional layers
        kernel_size: Size of the convolutional kernel
        nb_stacks: Number of residual stacks
        dilations: List of dilation factors
        dropout_rate: Dropout rate
        l2_reg: L2 regularization factor
        dense_units_list: List of units for dense layers
        activation: Activation function

    Returns:
        Keras model
    """
    # Input layers
    feature_input = Input(shape=(seq_length, n_features), name="feature_input")
    location_input = Input(shape=(seq_length,), dtype="int32", name="location_input")

    # Location embedding
    location_embedding = Embedding(
        input_dim=n_locations,
        output_dim=embedding_dim,
        embeddings_regularizer=l2(l2_reg),
        name="location_embedding",
    )(location_input)

    # Combine features with embedding
    combined_input = Concatenate(axis=2)([feature_input, location_embedding])
    x = BatchNormalization(name="input_norm")(combined_input)

    # Implementation of residual block for TCN
    def residual_block(x, dilation, nb_filters, kernel_size, block_idx, stack_idx):
        # First dilated convolution
        conv1 = Conv1D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            padding="causal",
            activation="linear",
            kernel_regularizer=l2(l2_reg),
            name=f"tcn_stack_{stack_idx}_block_{block_idx}_conv1",
        )(x)
        conv1 = BatchNormalization(name=f"tcn_stack_{stack_idx}_block_{block_idx}_bn1")(
            conv1
        )
        conv1 = Activation(activation)(conv1)
        conv1 = SpatialDropout1D(
            dropout_rate, name=f"tcn_stack_{stack_idx}_block_{block_idx}_dropout1"
        )(conv1)

        # Second dilated convolution
        conv2 = Conv1D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            padding="causal",
            activation="linear",
            kernel_regularizer=l2(l2_reg),
            name=f"tcn_stack_{stack_idx}_block_{block_idx}_conv2",
        )(conv1)
        conv2 = BatchNormalization(name=f"tcn_stack_{stack_idx}_block_{block_idx}_bn2")(
            conv2
        )
        conv2 = Activation(activation)(conv2)
        conv2 = SpatialDropout1D(
            dropout_rate, name=f"tcn_stack_{stack_idx}_block_{block_idx}_dropout2"
        )(conv2)

        # Skip connection
        if x.shape[-1] != nb_filters:
            x = Conv1D(
                filters=nb_filters,
                kernel_size=1,
                padding="same",
                kernel_regularizer=l2(l2_reg),
                name=f"tcn_stack_{stack_idx}_block_{block_idx}_skip",
            )(x)

        return Add(name=f"tcn_stack_{stack_idx}_block_{block_idx}_add")([x, conv2])

    # Build TCN model
    for stack_idx in range(nb_stacks):
        for block_idx, dilation in enumerate(dilations):
            x = residual_block(
                x, dilation, nb_filters, kernel_size, block_idx, stack_idx
            )

    # Global pooling
    x = GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Dense layers
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


# Dictionary mapping model names to their creation functions
MODEL_REGISTRY = {
    "lstm": create_lstm_model,
    "bilstm": create_bidirectional_lstm_model,
    "gru": create_gru_model,
    "bigru": create_bidirectional_gru_model,
    "cnn_lstm": create_cnn_lstm_model,
    "enhanced_cnn_lstm": create_enhanced_cnn_lstm_model,
    "transformer": create_transformer_model,
    "tcn": create_tcn_model,
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

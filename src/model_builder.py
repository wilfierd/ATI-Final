"""
LSTM Model Builder module for industrial pump predictive maintenance.

This module provides functions to build and compile LSTM-based neural network
models for multi-class classification of machine status.

Note: This module requires TensorFlow (pre-installed in Google Colab).
"""

from typing import Dict, List, Optional, Union

import numpy as np


def build_model(
    seq_length: int,
    n_features: int = 52,
    lstm_units: Optional[List[int]] = None,
    dropout_rate: float = 0.3,
    n_classes: int = 3,
    use_gru: bool = False
):
    """
    Build an LSTM/GRU model for multi-class classification.
    
    The model architecture consists of:
    - Input layer accepting sequences of shape (seq_length, n_features)
    - Stacked LSTM/GRU layers with dropout for regularization
    - Dense layer with ReLU activation
    - Output layer with softmax activation for 3-class classification
    
    Args:
        seq_length: Number of time steps in each input sequence.
        n_features: Number of sensor features (default 52).
        lstm_units: List of units for each LSTM layer (default [128, 64]).
        dropout_rate: Dropout rate for regularization (default 0.3).
        n_classes: Number of output classes (default 3).
        use_gru: If True, use GRU layers instead of LSTM (default False).
        
    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
        
    Raises:
        ImportError: If TensorFlow is not installed.
        ValueError: If invalid parameters are provided.
        
    Example:
        >>> model = build_model(seq_length=60, n_features=52)
        >>> model.summary()
        
    Requirements:
        - 3.1: Implement LSTM or GRU layers as core recurrent component
        - 3.2: Allow configurable hyperparameters
        - 3.4: Include regularization (dropout)
        - 3.5: Produce probability distributions using softmax
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            LSTM, GRU, Dense, Dropout, Input, BatchNormalization
        )
    except ImportError:
        raise ImportError(
            "TensorFlow is required for model building. "
            "In Google Colab, TensorFlow is pre-installed. "
            "For local development, install with: pip install tensorflow"
        )
    
    # Validate parameters
    if seq_length < 1:
        raise ValueError("seq_length must be at least 1")
    if n_features < 1:
        raise ValueError("n_features must be at least 1")
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be in range [0, 1)")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    
    # Default LSTM units if not provided
    if lstm_units is None:
        lstm_units = [128, 64]
    
    if len(lstm_units) < 1:
        raise ValueError("lstm_units must have at least one layer")
    
    # Select recurrent layer type
    RecurrentLayer = GRU if use_gru else LSTM
    layer_name = "GRU" if use_gru else "LSTM"
    
    # Build model
    model = Sequential(name=f"{layer_name}_Classifier")
    
    # Input layer
    model.add(Input(shape=(seq_length, n_features), name="input_layer"))
    
    # Add recurrent layers
    n_layers = len(lstm_units)
    for i, units in enumerate(lstm_units):
        # Return sequences for all but the last recurrent layer
        return_sequences = (i < n_layers - 1)
        
        model.add(RecurrentLayer(
            units=units,
            return_sequences=return_sequences,
            name=f"{layer_name.lower()}_{i+1}"
        ))
        
        # Add dropout after each recurrent layer
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate, name=f"dropout_{i+1}"))
    
    # Dense layer before output
    model.add(Dense(32, activation='relu', name="dense_1"))
    
    # Output layer with softmax for multi-class classification
    model.add(Dense(n_classes, activation='softmax', name="output"))
    
    return model


def compile_model(
    model,
    learning_rate: float = 0.001,
    loss: str = 'categorical_crossentropy',
    focal_loss_gamma: float = 2.0,
    focal_loss_alpha: Optional[List[float]] = None,
    clipnorm: float = 1.0
):
    """
    Compile the model with optimizer, loss function, and metrics.
    
    Args:
        model: Keras model to compile.
        learning_rate: Learning rate for Adam optimizer (default 0.001).
        loss: Loss function - 'categorical_crossentropy' or 'focal' (default 'categorical_crossentropy').
        focal_loss_gamma: Gamma parameter for focal loss (default 2.0).
        focal_loss_alpha: Alpha weights for focal loss (default None for balanced).
        clipnorm: Gradient clipping norm (default 1.0).
        
    Returns:
        tf.keras.Model: Compiled model.
        
    Raises:
        ImportError: If TensorFlow is not installed.
        ValueError: If invalid loss function is specified.
        
    Requirements:
        - 3.3: Use appropriate loss function (categorical cross-entropy or focal loss)
        - 4.3: Apply gradient clipping to prevent exploding gradients
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        raise ImportError(
            "TensorFlow is required for model compilation. "
            "In Google Colab, TensorFlow is pre-installed. "
            "For local development, install with: pip install tensorflow"
        )
    
    # Create optimizer with gradient clipping
    optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    
    # Select loss function
    if loss == 'categorical_crossentropy':
        loss_fn = 'categorical_crossentropy'
    elif loss == 'focal':
        # Import focal loss from imbalance_handler
        from imbalance_handler import get_focal_loss
        loss_fn = get_focal_loss(gamma=focal_loss_gamma, alpha=focal_loss_alpha)
    else:
        raise ValueError(
            f"Unsupported loss function: {loss}. "
            "Supported: 'categorical_crossentropy', 'focal'"
        )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model


def get_model_summary(model) -> str:
    """
    Get a string representation of the model summary.
    
    Args:
        model: Keras model.
        
    Returns:
        str: Model summary as string.
    """
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()


def create_lstm_model(
    seq_length: int,
    n_features: int = 52,
    lstm_units: Optional[List[int]] = None,
    dropout_rate: float = 0.3,
    n_classes: int = 3,
    learning_rate: float = 0.001,
    loss: str = 'categorical_crossentropy',
    focal_loss_gamma: float = 2.0,
    focal_loss_alpha: Optional[List[float]] = None,
    clipnorm: float = 1.0
):
    """
    Convenience function to build and compile an LSTM model in one step.
    
    This combines build_model() and compile_model() for simpler usage.
    
    Args:
        seq_length: Number of time steps in each input sequence.
        n_features: Number of sensor features (default 52).
        lstm_units: List of units for each LSTM layer (default [128, 64]).
        dropout_rate: Dropout rate for regularization (default 0.3).
        n_classes: Number of output classes (default 3).
        learning_rate: Learning rate for Adam optimizer (default 0.001).
        loss: Loss function - 'categorical_crossentropy' or 'focal'.
        focal_loss_gamma: Gamma parameter for focal loss (default 2.0).
        focal_loss_alpha: Alpha weights for focal loss.
        clipnorm: Gradient clipping norm (default 1.0).
        
    Returns:
        tf.keras.Model: Built and compiled Keras model.
        
    Example:
        >>> model = create_lstm_model(
        ...     seq_length=60,
        ...     n_features=52,
        ...     lstm_units=[128, 64],
        ...     dropout_rate=0.3,
        ...     loss='focal'
        ... )
    """
    model = build_model(
        seq_length=seq_length,
        n_features=n_features,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        n_classes=n_classes,
        use_gru=False
    )
    
    model = compile_model(
        model=model,
        learning_rate=learning_rate,
        loss=loss,
        focal_loss_gamma=focal_loss_gamma,
        focal_loss_alpha=focal_loss_alpha,
        clipnorm=clipnorm
    )
    
    return model

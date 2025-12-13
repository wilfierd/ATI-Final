"""
Training manager module for industrial pump predictive maintenance.

This module provides functions to manage model training with optimization
techniques including callbacks, learning rate scheduling, and early stopping.

Note: This module requires TensorFlow (pre-installed in Google Colab).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def get_callbacks(
    model_checkpoint_path: str = 'models/best_model.h5',
    patience: int = 10,
    lr_factor: float = 0.5,
    lr_patience: int = 5,
    min_lr: float = 1e-6,
    monitor: str = 'val_loss',
    verbose: int = 1
) -> List:
    """
    Create training callbacks for optimization.
    
    Returns a list of callbacks including:
    - EarlyStopping: Stops training when validation loss stops improving
    - ReduceLROnPlateau: Reduces learning rate when validation loss plateaus
    - ModelCheckpoint: Saves the best model during training
    
    Args:
        model_checkpoint_path: Path to save the best model.
        patience: Number of epochs with no improvement before early stopping.
        lr_factor: Factor by which to reduce learning rate.
        lr_patience: Number of epochs with no improvement before reducing LR.
        min_lr: Minimum learning rate.
        monitor: Metric to monitor for callbacks.
        verbose: Verbosity mode (0 = silent, 1 = progress bar).
        
    Returns:
        List of Keras callbacks.
        
    Raises:
        ImportError: If TensorFlow is not installed.
        
    Requirements:
        - 4.1: Implement optimization techniques (early stopping, LR scheduling)
        - 4.2: Reduce learning rate when validation loss plateaus
        - 4.5: Apply early stopping based on validation performance
    """
    try:
        from tensorflow.keras.callbacks import (
            EarlyStopping,
            ReduceLROnPlateau,
            ModelCheckpoint
        )
    except ImportError:
        raise ImportError(
            "TensorFlow is required for training callbacks. "
            "In Google Colab, TensorFlow is pre-installed. "
            "For local development, install with: pip install tensorflow"
        )
    
    callbacks = []
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=verbose,
        mode='min'
    )
    callbacks.append(early_stopping)
    
    # Learning rate reduction when validation loss plateaus
    lr_scheduler = ReduceLROnPlateau(
        monitor=monitor,
        factor=lr_factor,
        patience=lr_patience,
        min_lr=min_lr,
        verbose=verbose,
        mode='min'
    )
    callbacks.append(lr_scheduler)
    
    # Model checkpoint to save best model
    model_checkpoint = ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        verbose=verbose,
        mode='min'
    )
    callbacks.append(model_checkpoint)
    
    return callbacks



def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    class_weights: Optional[Dict[int, float]] = None,
    callbacks: Optional[List] = None,
    verbose: int = 1
):
    """
    Train the model with class weights support and callbacks.
    
    Tracks training and validation metrics per epoch through the
    returned History object.
    
    Args:
        model: Compiled Keras model to train.
        X_train: Training feature sequences of shape (n_samples, seq_length, n_features).
        y_train: Training labels (one-hot encoded).
        X_val: Validation feature sequences.
        y_val: Validation labels (one-hot encoded).
        epochs: Maximum number of training epochs.
        batch_size: Number of samples per gradient update.
        class_weights: Optional dictionary mapping class indices to weights.
        callbacks: Optional list of Keras callbacks. If None, uses default callbacks.
        verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
        
    Returns:
        tf.keras.callbacks.History: Training history containing metrics per epoch.
        
    Raises:
        ImportError: If TensorFlow is not installed.
        ValueError: If input shapes are incompatible.
        
    Requirements:
        - 4.4: Track training and validation metrics per epoch
        
    Example:
        >>> history = train(
        ...     model=model,
        ...     X_train=X_train, y_train=y_train,
        ...     X_val=X_val, y_val=y_val,
        ...     epochs=100,
        ...     batch_size=64,
        ...     class_weights=class_weights
        ... )
        >>> print(history.history['loss'])  # Training loss per epoch
        >>> print(history.history['val_loss'])  # Validation loss per epoch
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for model training. "
            "In Google Colab, TensorFlow is pre-installed. "
            "For local development, install with: pip install tensorflow"
        )
    
    # Validate input shapes
    if len(X_train.shape) != 3:
        raise ValueError(
            f"X_train must be 3D (samples, seq_length, features), "
            f"got shape {X_train.shape}"
        )
    if len(X_val.shape) != 3:
        raise ValueError(
            f"X_val must be 3D (samples, seq_length, features), "
            f"got shape {X_val.shape}"
        )
    
    # Use default callbacks if none provided
    if callbacks is None:
        callbacks = get_callbacks()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return history


def get_training_summary(history) -> Dict:
    """
    Extract summary statistics from training history.
    
    Args:
        history: Keras History object from model.fit().
        
    Returns:
        Dict containing:
            - final_train_loss: Final training loss
            - final_val_loss: Final validation loss
            - final_train_acc: Final training accuracy
            - final_val_acc: Final validation accuracy
            - best_val_loss: Best validation loss achieved
            - best_epoch: Epoch with best validation loss
            - total_epochs: Total number of epochs trained
    """
    hist = history.history
    
    # Find best epoch (lowest validation loss)
    val_losses = hist.get('val_loss', [])
    if val_losses:
        best_epoch = int(np.argmin(val_losses)) + 1
        best_val_loss = float(min(val_losses))
    else:
        best_epoch = len(hist.get('loss', []))
        best_val_loss = None
    
    return {
        'final_train_loss': float(hist['loss'][-1]) if hist.get('loss') else None,
        'final_val_loss': float(hist['val_loss'][-1]) if hist.get('val_loss') else None,
        'final_train_acc': float(hist['accuracy'][-1]) if hist.get('accuracy') else None,
        'final_val_acc': float(hist['val_accuracy'][-1]) if hist.get('val_accuracy') else None,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'total_epochs': len(hist.get('loss', []))
    }


def create_training_config(
    epochs: int = 100,
    batch_size: int = 64,
    early_stopping_patience: int = 10,
    lr_reduction_factor: float = 0.5,
    lr_reduction_patience: int = 5,
    min_lr: float = 1e-6,
    model_checkpoint_path: str = 'models/best_model.h5'
) -> Dict:
    """
    Create a configuration dictionary for training.
    
    This is a convenience function to bundle all training parameters
    into a single configuration object.
    
    Args:
        epochs: Maximum number of training epochs.
        batch_size: Number of samples per gradient update.
        early_stopping_patience: Epochs before early stopping.
        lr_reduction_factor: Factor to reduce learning rate.
        lr_reduction_patience: Epochs before reducing LR.
        min_lr: Minimum learning rate.
        model_checkpoint_path: Path to save best model.
        
    Returns:
        Dict containing all training configuration parameters.
    """
    return {
        'epochs': epochs,
        'batch_size': batch_size,
        'early_stopping_patience': early_stopping_patience,
        'lr_reduction_factor': lr_reduction_factor,
        'lr_reduction_patience': lr_reduction_patience,
        'min_lr': min_lr,
        'model_checkpoint_path': model_checkpoint_path
    }

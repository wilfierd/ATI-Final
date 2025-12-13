"""
Inference Engine Module for Industrial Pump Predictive Maintenance.

This module handles model saving, loading, and prediction on new sensor data.
It provides a complete inference pipeline for deploying the trained LSTM model.

Requirements: 6.1, 6.2, 6.3, 6.4
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pickle


def save_model(
    model,
    model_path: str,
    scaler=None,
    label_encoder=None,
    config: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Save the trained model and associated artifacts to disk.
    
    Saves the model in .h5 format and optionally saves the scaler,
    label encoder, and configuration as pickle files.
    
    Args:
        model: Trained Keras model to save.
        model_path: Path to save the model (.h5 file).
        scaler: Fitted StandardScaler for feature normalization.
        label_encoder: Fitted LabelEncoder for label decoding.
        config: Optional configuration dictionary with model parameters.
        
    Returns:
        Dictionary with paths to all saved artifacts.
        
    Raises:
        ValueError: If model_path doesn't end with .h5
        
    Requirements: 6.3
    """
    if not model_path.endswith('.h5'):
        raise ValueError("model_path must end with '.h5'")
    
    # Create directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    saved_paths = {}
    
    # Save the Keras model
    model.save(model_path)
    saved_paths['model'] = model_path
    
    # Determine base path for artifacts
    base_path = model_path.rsplit('.h5', 1)[0]
    
    # Save scaler if provided
    if scaler is not None:
        scaler_path = f"{base_path}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        saved_paths['scaler'] = scaler_path
    
    # Save label encoder if provided
    if label_encoder is not None:
        encoder_path = f"{base_path}_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        saved_paths['label_encoder'] = encoder_path
    
    # Save config if provided
    if config is not None:
        config_path = f"{base_path}_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        saved_paths['config'] = config_path
    
    return saved_paths



def load_model(
    model_path: str,
    load_scaler: bool = True,
    load_encoder: bool = True,
    load_config: bool = True,
    custom_objects: Optional[Dict] = None
) -> Dict:
    """
    Load a saved model and associated artifacts from disk.
    
    Args:
        model_path: Path to the saved model (.h5 file).
        load_scaler: Whether to load the scaler if available.
        load_encoder: Whether to load the label encoder if available.
        load_config: Whether to load the config if available.
        custom_objects: Custom objects for Keras model loading (e.g., focal loss).
        
    Returns:
        Dictionary containing:
            - 'model': Loaded Keras model
            - 'scaler': Loaded StandardScaler (if available and requested)
            - 'label_encoder': Loaded LabelEncoder (if available and requested)
            - 'config': Loaded configuration dict (if available and requested)
            
    Raises:
        FileNotFoundError: If model file doesn't exist.
        
    Requirements: 6.4
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model as keras_load_model
    except ImportError:
        raise ImportError(
            "TensorFlow is required for model loading. "
            "Install with: pip install tensorflow"
        )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    result = {}
    
    # Load the Keras model
    if custom_objects is None:
        # Try to import focal loss for custom objects
        try:
            from imbalance_handler import get_focal_loss
            custom_objects = {'focal_loss_fixed': get_focal_loss()}
        except ImportError:
            custom_objects = {}
    
    result['model'] = keras_load_model(model_path, custom_objects=custom_objects)
    
    # Determine base path for artifacts
    base_path = model_path.rsplit('.h5', 1)[0]
    
    # Load scaler if requested and available
    if load_scaler:
        scaler_path = f"{base_path}_scaler.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                result['scaler'] = pickle.load(f)
        else:
            result['scaler'] = None
    
    # Load label encoder if requested and available
    if load_encoder:
        encoder_path = f"{base_path}_encoder.pkl"
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                result['label_encoder'] = pickle.load(f)
        else:
            result['label_encoder'] = None
    
    # Load config if requested and available
    if load_config:
        config_path = f"{base_path}_config.pkl"
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                result['config'] = pickle.load(f)
        else:
            result['config'] = None
    
    return result


def preprocess_new_data(
    data: np.ndarray,
    scaler,
    seq_length: int,
    feature_columns: Optional[List[str]] = None
) -> np.ndarray:
    """
    Preprocess new sensor data for inference using the saved scaler.
    
    This function applies the same preprocessing pipeline used during training:
    1. Normalize features using the fitted scaler
    2. Create sequences for RNN input
    
    Args:
        data: New sensor data as numpy array of shape (n_samples, n_features)
              or pandas DataFrame.
        scaler: Fitted StandardScaler from training.
        seq_length: Sequence length used during training.
        feature_columns: Column names if data is a DataFrame.
        
    Returns:
        Preprocessed data ready for model prediction.
        Shape: (n_sequences, seq_length, n_features)
        
    Raises:
        ValueError: If data has insufficient samples for sequence creation.
        
    Requirements: 6.1
    """
    import pandas as pd
    
    # Convert DataFrame to numpy array if needed
    if isinstance(data, pd.DataFrame):
        if feature_columns is not None:
            data = data[feature_columns].values
        else:
            data = data.values
    
    # Ensure data is 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    
    # Check if we have enough samples
    if n_samples < seq_length:
        raise ValueError(
            f"Insufficient data for sequence creation. "
            f"Need at least {seq_length} samples, got {n_samples}."
        )
    
    # Normalize using the fitted scaler
    data_normalized = scaler.transform(data)
    
    # Create sequences
    n_sequences = n_samples - seq_length + 1
    sequences = np.zeros((n_sequences, seq_length, n_features))
    
    for i in range(n_sequences):
        sequences[i] = data_normalized[i:i + seq_length]
    
    return sequences



def predict(
    model,
    X: np.ndarray,
    label_encoder=None,
    return_probabilities: bool = True
) -> Dict[str, np.ndarray]:
    """
    Make predictions on preprocessed sensor data.
    
    Args:
        model: Trained Keras model.
        X: Preprocessed input data of shape (n_samples, seq_length, n_features).
        label_encoder: Optional LabelEncoder to decode predictions to class names.
        return_probabilities: Whether to return probability scores.
        
    Returns:
        Dictionary containing:
            - 'predicted_class': Array of predicted class indices
            - 'predicted_label': Array of predicted class names (if encoder provided)
            - 'confidence_scores': Array of confidence scores for each class
            - 'max_confidence': Array of maximum confidence for each prediction
            
    Requirements: 6.2
    """
    # Get probability predictions
    probabilities = model.predict(X, verbose=0)
    
    # Get predicted class indices
    predicted_indices = np.argmax(probabilities, axis=1)
    
    # Get maximum confidence scores
    max_confidence = np.max(probabilities, axis=1)
    
    result = {
        'predicted_class': predicted_indices,
        'max_confidence': max_confidence
    }
    
    # Add probability scores if requested
    if return_probabilities:
        result['confidence_scores'] = probabilities
    
    # Decode to class names if encoder provided
    if label_encoder is not None:
        result['predicted_label'] = label_encoder.inverse_transform(predicted_indices)
    
    return result


def predict_single(
    model,
    sensor_reading: np.ndarray,
    scaler,
    seq_length: int,
    label_encoder=None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Make a prediction on a single sequence of sensor readings.
    
    This is a convenience function for real-time inference on a single
    sequence of sensor data.
    
    Args:
        model: Trained Keras model.
        sensor_reading: Sensor data of shape (seq_length, n_features) or
                       (n_samples, n_features) where n_samples >= seq_length.
        scaler: Fitted StandardScaler from training.
        seq_length: Sequence length used during training.
        label_encoder: Optional LabelEncoder to decode predictions.
        class_names: Optional list of class names for output formatting.
        
    Returns:
        Dictionary containing:
            - 'predicted_class': Predicted class index
            - 'predicted_label': Predicted class name
            - 'confidence': Dictionary of confidence scores per class
            - 'max_confidence': Maximum confidence score
            
    Requirements: 6.2
    """
    # Default class names
    if class_names is None:
        class_names = ['NORMAL', 'RECOVERING', 'BROKEN']
    
    # Preprocess the data
    if len(sensor_reading.shape) == 2:
        if sensor_reading.shape[0] == seq_length:
            # Already a single sequence, just normalize
            data_normalized = scaler.transform(sensor_reading)
            X = data_normalized.reshape(1, seq_length, -1)
        else:
            # Multiple samples, create sequences
            X = preprocess_new_data(sensor_reading, scaler, seq_length)
            # Take the last sequence for single prediction
            X = X[-1:] if len(X) > 0 else X
    else:
        raise ValueError(
            f"Expected 2D array of shape (seq_length, n_features), "
            f"got shape {sensor_reading.shape}"
        )
    
    # Make prediction
    predictions = predict(model, X, label_encoder, return_probabilities=True)
    
    # Format output for single prediction
    probs = predictions['confidence_scores'][0]
    
    result = {
        'predicted_class': int(predictions['predicted_class'][0]),
        'predicted_label': class_names[predictions['predicted_class'][0]],
        'confidence': {name: float(probs[i]) for i, name in enumerate(class_names)},
        'max_confidence': float(predictions['max_confidence'][0])
    }
    
    if 'predicted_label' in predictions:
        result['predicted_label'] = predictions['predicted_label'][0]
    
    return result


def batch_inference(
    model,
    data: np.ndarray,
    scaler,
    seq_length: int,
    label_encoder=None,
    batch_size: int = 64,
    class_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Perform batch inference on a large dataset.
    
    Processes data in batches to handle memory constraints for large datasets.
    
    Args:
        model: Trained Keras model.
        data: Raw sensor data of shape (n_samples, n_features).
        scaler: Fitted StandardScaler from training.
        seq_length: Sequence length used during training.
        label_encoder: Optional LabelEncoder to decode predictions.
        batch_size: Batch size for inference.
        class_names: Optional list of class names.
        
    Returns:
        Dictionary containing predictions for all sequences.
        
    Requirements: 6.1, 6.2
    """
    # Default class names
    if class_names is None:
        class_names = ['NORMAL', 'RECOVERING', 'BROKEN']
    
    # Preprocess all data
    X = preprocess_new_data(data, scaler, seq_length)
    
    n_samples = len(X)
    all_predictions = []
    all_probabilities = []
    
    # Process in batches
    for i in range(0, n_samples, batch_size):
        batch = X[i:i + batch_size]
        probs = model.predict(batch, verbose=0)
        all_probabilities.append(probs)
        all_predictions.append(np.argmax(probs, axis=1))
    
    # Concatenate results
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    
    result = {
        'predicted_class': predictions,
        'confidence_scores': probabilities,
        'max_confidence': np.max(probabilities, axis=1)
    }
    
    # Decode to class names if encoder provided
    if label_encoder is not None:
        result['predicted_label'] = label_encoder.inverse_transform(predictions)
    else:
        result['predicted_label'] = np.array([class_names[i] for i in predictions])
    
    return result


def get_prediction_summary(predictions: Dict) -> Dict:
    """
    Generate a summary of batch predictions.
    
    Args:
        predictions: Output from predict() or batch_inference().
        
    Returns:
        Dictionary with prediction statistics.
    """
    pred_classes = predictions['predicted_class']
    confidences = predictions['max_confidence']
    
    unique, counts = np.unique(pred_classes, return_counts=True)
    
    summary = {
        'total_predictions': len(pred_classes),
        'class_distribution': dict(zip(unique.tolist(), counts.tolist())),
        'mean_confidence': float(np.mean(confidences)),
        'min_confidence': float(np.min(confidences)),
        'max_confidence': float(np.max(confidences)),
        'std_confidence': float(np.std(confidences))
    }
    
    # Add percentage distribution
    summary['class_percentages'] = {
        k: v / len(pred_classes) * 100 
        for k, v in summary['class_distribution'].items()
    }
    
    return summary

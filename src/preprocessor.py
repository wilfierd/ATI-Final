"""
Data preprocessing module for industrial pump predictive maintenance.

This module handles data cleaning, normalization, sequence creation,
label encoding, and train/validation/test splitting for RNN model training.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'ffill',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in sensor data using forward fill.
    
    Args:
        df: Input DataFrame with potential missing values.
        method: Imputation method ('ffill' for forward fill).
        columns: Specific columns to impute. If None, imputes all numeric columns.
        
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
        
    Raises:
        ValueError: If an unsupported imputation method is specified.
    """
    if method not in ['ffill', 'bfill', 'interpolate', 'mean']:
        raise ValueError(
            f"Unsupported imputation method: {method}. "
            "Supported methods: 'ffill', 'bfill', 'interpolate', 'mean'"
        )
    
    df_copy = df.copy()
    
    # Determine columns to impute
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'ffill':
        df_copy[columns] = df_copy[columns].ffill()
        # Handle any remaining NaNs at the start with backward fill
        df_copy[columns] = df_copy[columns].bfill()
    elif method == 'bfill':
        df_copy[columns] = df_copy[columns].bfill()
        df_copy[columns] = df_copy[columns].ffill()
    elif method == 'interpolate':
        df_copy[columns] = df_copy[columns].interpolate(method='linear')
        df_copy[columns] = df_copy[columns].bfill().ffill()
    elif method == 'mean':
        for col in columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    
    return df_copy



def normalize_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize sensor features using StandardScaler.
    
    Args:
        df: Input DataFrame with sensor features.
        feature_columns: List of column names to normalize.
        scaler: Pre-fitted scaler for inference. If None, fits a new scaler.
        
    Returns:
        Tuple containing:
            - pd.DataFrame: DataFrame with normalized features.
            - StandardScaler: Fitted scaler for later use in inference.
    """
    df_copy = df.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        df_copy[feature_columns] = scaler.fit_transform(df_copy[feature_columns])
    else:
        df_copy[feature_columns] = scaler.transform(df_copy[feature_columns])
    
    return df_copy, scaler


def create_sequences(
    data: np.ndarray,
    seq_length: int
) -> np.ndarray:
    """
    Create sliding window sequences for RNN input.
    
    Args:
        data: Input data array of shape (n_samples, n_features).
        seq_length: Number of time steps in each sequence.
        
    Returns:
        np.ndarray: 3D array of shape (n_sequences, seq_length, n_features).
        
    Raises:
        ValueError: If sequence length is greater than data length.
    """
    if seq_length > len(data):
        raise ValueError(
            f"Sequence length ({seq_length}) cannot be greater than "
            f"data length ({len(data)})"
        )
    
    if seq_length < 1:
        raise ValueError("Sequence length must be at least 1")
    
    n_samples = len(data) - seq_length + 1
    n_features = data.shape[1] if len(data.shape) > 1 else 1
    
    # Handle 1D input
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    sequences = np.zeros((n_samples, seq_length, n_features))
    
    for i in range(n_samples):
        sequences[i] = data[i:i + seq_length]
    
    return sequences


def create_sequences_with_labels(
    features: np.ndarray,
    labels: np.ndarray,
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences with corresponding labels (using last label in sequence).
    
    Args:
        features: Feature array of shape (n_samples, n_features).
        labels: Label array of shape (n_samples,) or (n_samples, n_classes).
        seq_length: Number of time steps in each sequence.
        
    Returns:
        Tuple containing:
            - np.ndarray: Sequences of shape (n_sequences, seq_length, n_features).
            - np.ndarray: Labels corresponding to each sequence.
    """
    X = create_sequences(features, seq_length)
    # Use the label at the end of each sequence
    y = labels[seq_length - 1:]
    
    return X, y



def encode_labels(
    labels: pd.Series,
    encoder: Optional[LabelEncoder] = None,
    class_order: Optional[List[str]] = None
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Convert machine_status to numerical labels.
    
    Args:
        labels: Series containing machine_status values.
        encoder: Pre-fitted encoder for inference. If None, fits a new encoder.
        class_order: Optional list specifying class order [NORMAL, RECOVERING, BROKEN].
        
    Returns:
        Tuple containing:
            - np.ndarray: Encoded numerical labels.
            - LabelEncoder: Fitted encoder for later use.
    """
    if encoder is None:
        encoder = LabelEncoder()
        if class_order is not None:
            encoder.fit(class_order)
            encoded = encoder.transform(labels)
        else:
            encoded = encoder.fit_transform(labels)
    else:
        encoded = encoder.transform(labels)
    
    return encoded, encoder


def decode_labels(
    encoded_labels: np.ndarray,
    encoder: LabelEncoder
) -> np.ndarray:
    """
    Convert numerical labels back to original string labels.
    
    Args:
        encoded_labels: Array of numerical labels.
        encoder: Fitted LabelEncoder used for encoding.
        
    Returns:
        np.ndarray: Original string labels.
    """
    return encoder.inverse_transform(encoded_labels)


def one_hot_encode(
    labels: np.ndarray,
    n_classes: int = 3
) -> np.ndarray:
    """
    Convert numerical labels to one-hot encoded format.
    
    Args:
        labels: Array of numerical labels (0, 1, 2, ...).
        n_classes: Number of classes.
        
    Returns:
        np.ndarray: One-hot encoded labels of shape (n_samples, n_classes).
    """
    n_samples = len(labels)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), labels.astype(int)] = 1
    return one_hot


def one_hot_decode(
    one_hot_labels: np.ndarray
) -> np.ndarray:
    """
    Convert one-hot encoded labels back to numerical labels.
    
    Args:
        one_hot_labels: One-hot encoded array of shape (n_samples, n_classes).
        
    Returns:
        np.ndarray: Numerical labels.
    """
    return np.argmax(one_hot_labels, axis=1)



def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets with stratification.
    
    Args:
        X: Feature array.
        y: Label array (can be one-hot encoded or numerical).
        val_size: Proportion of data for validation (default 0.15).
        test_size: Proportion of data for test (default 0.15).
        random_state: Random seed for reproducibility.
        stratify: Whether to preserve class distribution in splits.
        
    Returns:
        Tuple containing:
            - X_train, X_val, X_test: Feature arrays for each split.
            - y_train, y_val, y_test: Label arrays for each split.
    """
    # Convert one-hot to numerical for stratification if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        y_stratify = np.argmax(y, axis=1)
    else:
        y_stratify = y
    
    # Calculate split ratios
    # First split: separate test set
    train_val_size = 1.0 - test_size
    
    stratify_labels = y_stratify if stratify else None
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )
    
    # Second split: separate validation from training
    # Adjust val_size relative to remaining data
    val_size_adjusted = val_size / train_val_size
    
    if stratify:
        if len(y_train_val.shape) > 1 and y_train_val.shape[1] > 1:
            stratify_labels_2 = np.argmax(y_train_val, axis=1)
        else:
            stratify_labels_2 = y_train_val
    else:
        stratify_labels_2 = None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_labels_2
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_pipeline(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    seq_length: int = 60,
    val_size: float = 0.15,
    test_size: float = 0.15,
    class_order: Optional[List[str]] = None,
    random_state: int = 42
) -> Dict:
    """
    Complete preprocessing pipeline for sensor data.
    
    Args:
        df: Raw sensor DataFrame.
        feature_columns: List of sensor column names.
        target_column: Name of target column.
        seq_length: Sequence length for RNN input.
        val_size: Validation set proportion.
        test_size: Test set proportion.
        class_order: Optional class ordering for label encoding.
        random_state: Random seed for reproducibility.
        
    Returns:
        Dict containing:
            - X_train, X_val, X_test: Feature sequences.
            - y_train, y_val, y_test: One-hot encoded labels.
            - scaler: Fitted StandardScaler.
            - label_encoder: Fitted LabelEncoder.
    """
    # Step 1: Handle missing values
    df_clean = handle_missing_values(df, method='ffill', columns=feature_columns)
    
    # Step 2: Normalize features
    df_normalized, scaler = normalize_features(df_clean, feature_columns)
    
    # Step 3: Encode labels
    labels_encoded, label_encoder = encode_labels(
        df_normalized[target_column],
        class_order=class_order
    )
    
    # Step 4: Create sequences
    features = df_normalized[feature_columns].values
    X_seq, y_seq = create_sequences_with_labels(features, labels_encoded, seq_length)
    
    # Step 5: One-hot encode labels
    y_onehot = one_hot_encode(y_seq, n_classes=len(label_encoder.classes_))
    
    # Step 6: Split data
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X_seq, y_onehot,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoder': label_encoder
    }

"""
Data loading module for industrial pump predictive maintenance.

This module handles loading and parsing of sensor data from CSV files,
with support for both local files and Google Colab with Google Drive.
"""

import os
from typing import List, Optional

import pandas as pd


def mount_google_drive() -> str:
    """
    Mount Google Drive in Google Colab environment.
    
    Returns:
        str: Path to the mounted Google Drive root.
        
    Raises:
        RuntimeError: If not running in Google Colab environment.
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return '/content/drive/MyDrive'
    except ImportError:
        raise RuntimeError(
            "Google Colab environment not detected. "
            "Use load_csv() with a local file path instead."
        )


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load sensor data from CSV file.
    
    Args:
        file_path: Path to the sensor.csv file.
        
    Returns:
        pd.DataFrame: DataFrame containing sensor data with columns:
            - timestamp: datetime of sensor reading
            - sensor_00 to sensor_51: 52 sensor measurements
            - machine_status: target variable (NORMAL, BROKEN, RECOVERING)
            
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sensor data file not found: {file_path}")
    
    # Load CSV, handling the unnamed index column
    df = pd.read_csv(file_path, index_col=0)
    
    # Parse timestamp column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Validate required columns exist
    feature_cols = get_feature_columns()
    target_col = get_target_column()
    
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing sensor columns in CSV: {missing_features[:5]}... "
            f"({len(missing_features)} total)"
        )
    
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in CSV")
    
    return df


def get_feature_columns() -> List[str]:
    """
    Get list of sensor feature column names.
    
    Returns:
        List[str]: List of 52 sensor column names (sensor_00 to sensor_51).
    """
    return [f'sensor_{i:02d}' for i in range(52)]


def get_target_column() -> str:
    """
    Get the target column name.
    
    Returns:
        str: Name of the target column ('machine_status').
    """
    return 'machine_status'


def get_class_names() -> List[str]:
    """
    Get the class names for machine status.
    
    Returns:
        List[str]: List of class names in order [NORMAL, RECOVERING, BROKEN].
    """
    return ['NORMAL', 'RECOVERING', 'BROKEN']


def load_from_drive(
    filename: str = 'sensor.csv',
    drive_folder: Optional[str] = None
) -> pd.DataFrame:
    """
    Load sensor data from Google Drive (for Colab usage).
    
    Args:
        filename: Name of the CSV file.
        drive_folder: Optional subfolder path within Google Drive.
        
    Returns:
        pd.DataFrame: Loaded sensor data.
    """
    drive_root = mount_google_drive()
    
    if drive_folder:
        file_path = os.path.join(drive_root, drive_folder, filename)
    else:
        file_path = os.path.join(drive_root, filename)
    
    return load_csv(file_path)

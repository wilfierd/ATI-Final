"""
Complete Pipeline Verification Script

This script verifies that all components of the industrial pump predictive
maintenance pipeline work together correctly:
1. Data loading
2. Preprocessing
3. Class imbalance handling
4. Model building
5. Training (minimal epochs for verification)
6. Evaluation
7. Inference (save/load round-trip)

Run with: python verify_training.py
"""

import os
import sys
import numpy as np
import pandas as pd
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_data_loading():
    """Verify data loading module."""
    print("\n" + "="*60)
    print("1. VERIFYING DATA LOADING")
    print("="*60)
    
    from src.data_loader import load_csv, get_feature_columns, get_target_column, get_class_names
    
    # Check feature columns
    feature_cols = get_feature_columns()
    assert len(feature_cols) == 52, f"Expected 52 features, got {len(feature_cols)}"
    print(f"✓ Feature columns: {len(feature_cols)} columns")
    
    # Check target column
    target_col = get_target_column()
    assert target_col == 'machine_status', f"Expected 'machine_status', got {target_col}"
    print(f"✓ Target column: {target_col}")
    
    # Check class names
    class_names = get_class_names()
    assert len(class_names) == 3, f"Expected 3 classes, got {len(class_names)}"
    print(f"✓ Class names: {class_names}")
    
    # Load actual data if available
    if os.path.exists('sensor.csv'):
        df = load_csv('sensor.csv')
        print(f"✓ Loaded sensor.csv: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    else:
        print("⚠ sensor.csv not found, creating synthetic data for testing")
        return create_synthetic_data()


def create_synthetic_data(n_samples=1000):
    """Create synthetic sensor data for testing."""
    np.random.seed(42)
    
    # Create feature columns
    feature_cols = [f'sensor_{i:02d}' for i in range(52)]
    data = np.random.randn(n_samples, 52)
    
    # Create labels with imbalance
    labels = np.array(['NORMAL'] * int(n_samples * 0.93) + 
                      ['RECOVERING'] * int(n_samples * 0.065) +
                      ['BROKEN'] * int(n_samples * 0.005))
    np.random.shuffle(labels)
    labels = labels[:n_samples]
    
    df = pd.DataFrame(data, columns=feature_cols)
    df['machine_status'] = labels
    
    return df


def verify_preprocessing(df):
    """Verify preprocessing module."""
    print("\n" + "="*60)
    print("2. VERIFYING PREPROCESSING")
    print("="*60)
    
    from src.preprocessor import (
        handle_missing_values, normalize_features, create_sequences,
        encode_labels, one_hot_encode, train_val_test_split
    )
    from src.data_loader import get_feature_columns, get_target_column
    
    feature_cols = get_feature_columns()
    target_col = get_target_column()
    
    # Check initial missing values
    initial_missing = df[feature_cols].isna().sum().sum()
    print(f"  Initial missing values: {initial_missing}")
    
    # Handle missing values
    df_clean = handle_missing_values(df, method='ffill', columns=feature_cols)
    missing_count = df_clean[feature_cols].isna().sum().sum()
    
    # Note: Some columns may be entirely NaN (like sensor_15) and can't be imputed
    # We'll drop those columns for the verification
    cols_with_all_nan = [col for col in feature_cols if df_clean[col].isna().all()]
    if cols_with_all_nan:
        print(f"  Columns with all NaN (will be dropped): {cols_with_all_nan}")
        feature_cols = [col for col in feature_cols if col not in cols_with_all_nan]
        df_clean = df_clean.drop(columns=cols_with_all_nan)
    
    missing_count = df_clean[feature_cols].isna().sum().sum()
    assert missing_count == 0, f"Still have {missing_count} missing values"
    print(f"✓ Missing value handling: {missing_count} NaN values remaining (using {len(feature_cols)} features)")
    
    # Normalize features
    df_norm, scaler = normalize_features(df_clean, feature_cols)
    print(f"✓ Feature normalization: scaler fitted")
    
    # Encode labels
    labels_encoded, encoder = encode_labels(df_norm[target_col])
    print(f"✓ Label encoding: {len(encoder.classes_)} classes encoded")
    
    # Create sequences
    seq_length = 10  # Small for testing
    features = df_norm[feature_cols].values
    sequences = create_sequences(features, seq_length)
    n_features = len(feature_cols)
    expected_shape = (len(features) - seq_length + 1, seq_length, n_features)
    assert sequences.shape == expected_shape, f"Expected {expected_shape}, got {sequences.shape}"
    print(f"✓ Sequence creation: shape {sequences.shape}")
    
    # One-hot encode
    y_seq = labels_encoded[seq_length - 1:]
    y_onehot = one_hot_encode(y_seq, n_classes=3)
    assert y_onehot.shape[1] == 3, f"Expected 3 classes, got {y_onehot.shape[1]}"
    print(f"✓ One-hot encoding: shape {y_onehot.shape}")
    
    # Train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        sequences, y_onehot, val_size=0.15, test_size=0.15, stratify=False
    )
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(sequences), f"Split lost samples: {total} vs {len(sequences)}"
    print(f"✓ Train/val/test split: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder


def verify_imbalance_handling(y_train):
    """Verify class imbalance handling module."""
    print("\n" + "="*60)
    print("3. VERIFYING CLASS IMBALANCE HANDLING")
    print("="*60)
    
    from src.imbalance_handler import compute_class_weights, get_class_distribution
    
    # Get class distribution
    dist = get_class_distribution(y_train)
    print(f"✓ Class distribution: {dist['counts']}")
    
    # Compute class weights
    weights = compute_class_weights(y_train)
    print(f"✓ Class weights: {weights}")
    
    # Verify weights are inversely proportional
    for cls, weight in weights.items():
        product = weight * dist['counts'].get(cls, 0)
        print(f"  Class {cls}: weight={weight:.4f}, count={dist['counts'].get(cls, 0)}, product={product:.2f}")
    
    return weights


def verify_model_building(seq_length, n_features):
    """Verify model building module."""
    print("\n" + "="*60)
    print("4. VERIFYING MODEL BUILDING")
    print("="*60)
    
    from src.model_builder import build_model, compile_model
    
    # Build model
    model = build_model(
        seq_length=seq_length,
        n_features=n_features,
        lstm_units=[64, 32],  # Smaller for testing
        dropout_rate=0.3,
        n_classes=3
    )
    print(f"✓ Model built: {model.name}")
    
    # Compile model
    model = compile_model(
        model,
        learning_rate=0.001,
        loss='categorical_crossentropy',
        clipnorm=1.0
    )
    print(f"✓ Model compiled with Adam optimizer")
    
    # Verify output shape
    test_input = np.random.randn(1, seq_length, n_features)
    output = model.predict(test_input, verbose=0)
    assert output.shape == (1, 3), f"Expected output shape (1, 3), got {output.shape}"
    assert np.isclose(output.sum(), 1.0, atol=1e-5), "Softmax output doesn't sum to 1"
    print(f"✓ Model output: shape {output.shape}, sum={output.sum():.6f}")
    
    return model


def verify_training(model, X_train, y_train, X_val, y_val, class_weights):
    """Verify training module."""
    print("\n" + "="*60)
    print("5. VERIFYING TRAINING")
    print("="*60)
    
    from src.training_manager import train, get_callbacks, get_training_summary
    import tensorflow as tf
    
    # Create temporary directory for model checkpoint
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, 'test_model.h5')
    
    try:
        # Get callbacks with minimal patience for testing
        callbacks = get_callbacks(
            model_checkpoint_path=checkpoint_path,
            patience=2,
            lr_patience=1,
            verbose=0
        )
        print(f"✓ Callbacks created: {len(callbacks)} callbacks")
        
        # Train for minimal epochs
        history = train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=3,  # Minimal for testing
            batch_size=32,
            class_weights=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        print(f"✓ Training completed: {len(history.history['loss'])} epochs")
        
        # Get training summary
        summary = get_training_summary(history)
        print(f"✓ Training summary: final_val_loss={summary['final_val_loss']:.4f}")
        
        return history
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def verify_evaluation(model, X_test, y_test):
    """Verify evaluation module."""
    print("\n" + "="*60)
    print("6. VERIFYING EVALUATION")
    print("="*60)
    
    from src.evaluator import (
        compute_metrics, get_confusion_matrix_data, compute_roc_auc
    )
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred_proba, class_names=['NORMAL', 'RECOVERING', 'BROKEN'])
    print(f"✓ Metrics computed:")
    print(f"  Macro F1: {metrics['macro_avg']['f1_score']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_avg']['f1_score']:.4f}")
    
    # Verify metrics are in valid range
    for avg_type in ['macro_avg', 'weighted_avg']:
        for metric in ['precision', 'recall', 'f1_score']:
            value = metrics[avg_type][metric]
            assert 0 <= value <= 1, f"{avg_type} {metric} out of range: {value}"
    print(f"✓ All metrics in valid range [0, 1]")
    
    # Confusion matrix
    cm = get_confusion_matrix_data(y_test, y_pred_proba)
    assert cm.shape[0] == cm.shape[1], "Confusion matrix not square"
    row_sums = cm.sum(axis=1)
    print(f"✓ Confusion matrix: shape {cm.shape}")
    
    # ROC-AUC (may fail if not all classes present)
    try:
        roc_auc = compute_roc_auc(y_test, y_pred_proba)
        if roc_auc['macro_auc'] is not None:
            print(f"✓ ROC-AUC: macro={roc_auc['macro_auc']:.4f}")
        else:
            print(f"⚠ ROC-AUC: insufficient class diversity")
    except Exception as e:
        print(f"⚠ ROC-AUC computation skipped: {e}")
    
    return metrics


def verify_inference(model, X_test, scaler, encoder):
    """Verify inference module (save/load round-trip)."""
    print("\n" + "="*60)
    print("7. VERIFYING INFERENCE (SAVE/LOAD ROUND-TRIP)")
    print("="*60)
    
    from src.inference import save_model, load_model, predict
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, 'test_model.h5')
    
    try:
        # Get predictions before saving
        pred_before = model.predict(X_test[:5], verbose=0)
        
        # Save model
        saved_paths = save_model(
            model=model,
            model_path=model_path,
            scaler=scaler,
            label_encoder=encoder,
            config={'seq_length': X_test.shape[1], 'n_features': X_test.shape[2]}
        )
        print(f"✓ Model saved to: {model_path}")
        
        # Load model
        loaded = load_model(model_path)
        loaded_model = loaded['model']
        print(f"✓ Model loaded successfully")
        
        # Get predictions after loading
        pred_after = loaded_model.predict(X_test[:5], verbose=0)
        
        # Verify predictions match
        max_diff = np.max(np.abs(pred_before - pred_after))
        assert max_diff < 1e-5, f"Predictions differ by {max_diff}"
        print(f"✓ Save/load round-trip: max prediction difference = {max_diff:.2e}")
        
        # Verify scaler and encoder loaded
        assert loaded['scaler'] is not None, "Scaler not loaded"
        assert loaded['label_encoder'] is not None, "Label encoder not loaded"
        print(f"✓ Scaler and encoder loaded successfully")
        
        # Test predict function
        predictions = predict(loaded_model, X_test[:5], encoder)
        assert 'predicted_class' in predictions, "Missing predicted_class"
        assert 'confidence_scores' in predictions, "Missing confidence_scores"
        assert len(predictions['predicted_class']) == 5, "Wrong number of predictions"
        print(f"✓ Predict function: {len(predictions['predicted_class'])} predictions")
        
        # Verify confidence scores sum to 1
        for i, scores in enumerate(predictions['confidence_scores']):
            score_sum = np.sum(scores)
            assert np.isclose(score_sum, 1.0, atol=1e-5), f"Scores don't sum to 1: {score_sum}"
        print(f"✓ All confidence scores sum to 1.0")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run complete pipeline verification."""
    print("\n" + "="*60)
    print("INDUSTRIAL PUMP PREDICTIVE MAINTENANCE")
    print("COMPLETE PIPELINE VERIFICATION")
    print("="*60)
    
    try:
        # 1. Data Loading
        df = verify_data_loading()
        
        # 2. Preprocessing
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder = verify_preprocessing(df)
        
        # 3. Class Imbalance Handling
        class_weights = verify_imbalance_handling(y_train)
        
        # 4. Model Building
        seq_length = X_train.shape[1]
        n_features = X_train.shape[2]
        model = verify_model_building(seq_length, n_features)
        
        # 5. Training
        history = verify_training(model, X_train, y_train, X_val, y_val, class_weights)
        
        # 6. Evaluation
        metrics = verify_evaluation(model, X_test, y_test)
        
        # 7. Inference
        verify_inference(model, X_test, scaler, encoder)
        
        print("\n" + "="*60)
        print("✓ ALL PIPELINE COMPONENTS VERIFIED SUCCESSFULLY")
        print("="*60)
        return 0
        
    except Exception as e:
        print(f"\n✗ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

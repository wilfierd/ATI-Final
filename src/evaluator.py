"""
Model Evaluator Module

Provides functions for evaluating model performance with imbalanced-aware metrics.
Includes precision, recall, F1-score computation, confusion matrix generation,
training curves visualization, and ROC-AUC computation.

Requirements: 5.1, 5.2, 5.3, 5.4, 7.1
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute precision, recall, F1 for each class and macro/weighted averages.
    
    Args:
        y_true: Ground truth labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        class_names: Optional list of class names for the report
        
    Returns:
        Dictionary containing per-class and aggregate metrics
        
    Requirements: 5.1, 5.3
    """
    # Convert one-hot to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true.flatten()
        
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = y_pred.flatten()
    
    # Default class names
    if class_names is None:
        n_classes = len(np.unique(np.concatenate([y_true_indices, y_pred_indices])))
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    n_classes = len(class_names)
    
    # Compute per-class metrics
    per_class_metrics = {}
    for i, name in enumerate(class_names):
        # Binary mask for this class
        y_true_binary = (y_true_indices == i).astype(int)
        y_pred_binary = (y_pred_indices == i).astype(int)
        
        # Compute metrics (handle zero division)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Support (number of true instances)
        support = int(np.sum(y_true_binary))
        
        per_class_metrics[name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': support
        }
    
    # Compute macro averages (unweighted mean of per-class metrics)
    macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
    macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
    macro_f1 = np.mean([m['f1_score'] for m in per_class_metrics.values()])
    
    # Compute weighted averages (weighted by support)
    total_support = sum(m['support'] for m in per_class_metrics.values())
    if total_support > 0:
        weighted_precision = sum(
            m['precision'] * m['support'] for m in per_class_metrics.values()
        ) / total_support
        weighted_recall = sum(
            m['recall'] * m['support'] for m in per_class_metrics.values()
        ) / total_support
        weighted_f1 = sum(
            m['f1_score'] * m['support'] for m in per_class_metrics.values()
        ) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    
    return {
        'per_class': per_class_metrics,
        'macro_avg': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1_score': float(macro_f1)
        },
        'weighted_avg': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1_score': float(weighted_f1)
        },
        'total_support': total_support
    }



def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    title: str = 'Confusion Matrix'
) -> plt.Figure:
    """
    Generate confusion matrix as a heatmap.
    
    Args:
        y_true: Ground truth labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        class_names: Optional list of class names
        normalize: If True, normalize by row (true labels)
        figsize: Figure size tuple
        cmap: Colormap for heatmap
        title: Plot title
        
    Returns:
        matplotlib Figure object
        
    Requirements: 5.2
    """
    # Convert one-hot to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true.flatten()
        
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = y_pred.flatten()
    
    # Default class names
    if class_names is None:
        n_classes = len(np.unique(np.concatenate([y_true_indices, y_pred_indices])))
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    
    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)  # Handle division by zero
        fmt = '.2%'
    else:
        cm_display = cm
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig


def get_confusion_matrix_data(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Get raw confusion matrix data without plotting.
    
    Args:
        y_true: Ground truth labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        
    Returns:
        Confusion matrix as numpy array
        
    Requirements: 5.2
    """
    # Convert one-hot to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true.flatten()
        
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = y_pred.flatten()
    
    return confusion_matrix(y_true_indices, y_pred_indices)


def plot_training_curves(
    history: Any,
    figsize: Tuple[int, int] = (14, 5),
    title_prefix: str = ''
) -> plt.Figure:
    """
    Plot training/validation loss and accuracy curves.
    
    Args:
        history: Keras History object or dict with 'loss', 'val_loss', 
                 'accuracy', 'val_accuracy' keys
        figsize: Figure size tuple
        title_prefix: Optional prefix for plot titles
        
    Returns:
        matplotlib Figure object
        
    Requirements: 7.1
    """
    # Handle both History object and dict
    if hasattr(history, 'history'):
        hist_dict = history.history
    else:
        hist_dict = history
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss curves
    ax1 = axes[0]
    epochs = range(1, len(hist_dict['loss']) + 1)
    
    ax1.plot(epochs, hist_dict['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in hist_dict:
        ax1.plot(epochs, hist_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        # Mark best epoch
        best_epoch = np.argmin(hist_dict['val_loss']) + 1
        best_val_loss = min(hist_dict['val_loss'])
        ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7,
                    label=f'Best Epoch ({best_epoch})')
        ax1.scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{title_prefix}Training and Validation Loss'.strip(), fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2 = axes[1]
    
    if 'accuracy' in hist_dict:
        ax2.plot(epochs, hist_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in hist_dict:
        ax2.plot(epochs, hist_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        
        # Mark best epoch (based on val_loss)
        if 'val_loss' in hist_dict:
            best_epoch = np.argmin(hist_dict['val_loss']) + 1
            ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7,
                        label=f'Best Epoch ({best_epoch})')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'{title_prefix}Training and Validation Accuracy'.strip(), fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compute_roc_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute multi-class ROC-AUC score.
    
    Args:
        y_true: Ground truth labels (one-hot encoded or class indices)
        y_pred_proba: Predicted probabilities for each class (n_samples, n_classes)
        average: Averaging method ('macro', 'weighted', or 'micro')
        
    Returns:
        Dictionary with ROC-AUC scores
        
    Requirements: 5.4
    """
    # Ensure y_pred_proba is 2D
    if len(y_pred_proba.shape) == 1:
        raise ValueError("y_pred_proba must be 2D array with shape (n_samples, n_classes)")
    
    n_classes = y_pred_proba.shape[1]
    
    # Convert y_true to one-hot if needed
    if len(y_true.shape) == 1 or y_true.shape[1] == 1:
        y_true_flat = y_true.flatten().astype(int)
        y_true_onehot = np.zeros((len(y_true_flat), n_classes))
        y_true_onehot[np.arange(len(y_true_flat)), y_true_flat] = 1
    else:
        y_true_onehot = y_true
    
    result = {}
    
    # Compute per-class ROC-AUC
    per_class_auc = []
    for i in range(n_classes):
        try:
            # Check if class has both positive and negative samples
            if len(np.unique(y_true_onehot[:, i])) > 1:
                auc = roc_auc_score(y_true_onehot[:, i], y_pred_proba[:, i])
                per_class_auc.append(auc)
                result[f'class_{i}_auc'] = float(auc)
            else:
                # Class not present in y_true, skip
                result[f'class_{i}_auc'] = None
        except ValueError:
            result[f'class_{i}_auc'] = None
    
    # Compute overall ROC-AUC using OvR (One-vs-Rest)
    try:
        result['macro_auc'] = float(roc_auc_score(
            y_true_onehot, y_pred_proba, average='macro', multi_class='ovr'
        ))
    except ValueError:
        result['macro_auc'] = None
    
    try:
        result['weighted_auc'] = float(roc_auc_score(
            y_true_onehot, y_pred_proba, average='weighted', multi_class='ovr'
        ))
    except ValueError:
        result['weighted_auc'] = None
    
    return result


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Generate and print a formatted classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        
    Returns:
        Classification report as string
    """
    # Convert one-hot to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true.flatten()
        
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = y_pred.flatten()
    
    report = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=class_names,
        zero_division=0
    )
    
    return report


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation combining all metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels (one-hot encoded)
        class_names: Optional list of class names
        verbose: Whether to print results
        
    Returns:
        Dictionary with all evaluation results
    """
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = y_pred_proba  # Keep probabilities for metrics
    
    # Compute all metrics
    metrics = compute_metrics(y_test, y_pred, class_names)
    roc_auc = compute_roc_auc(y_test, y_pred_proba)
    cm = get_confusion_matrix_data(y_test, y_pred)
    
    results = {
        'metrics': metrics,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred_proba': y_pred_proba,
        'y_pred': np.argmax(y_pred_proba, axis=1)
    }
    
    if verbose:
        print("=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        
        print("\nClassification Report:")
        print(print_classification_report(y_test, y_pred, class_names))
        
        print("\nROC-AUC Scores:")
        for key, value in roc_auc.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: N/A (insufficient samples)")
        
        print("\nConfusion Matrix:")
        print(cm)
    
    return results

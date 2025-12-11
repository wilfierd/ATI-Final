"""
Class imbalance handling module for industrial pump predictive maintenance.

This module provides techniques to handle extreme class imbalance in the dataset,
including class weight computation and focal loss implementation.

Note: Focal loss requires TensorFlow (pre-installed in Google Colab).
For local testing without TensorFlow, only compute_class_weights and 
get_class_distribution are available.
"""

from typing import Dict, List, Optional, Union

import numpy as np


def compute_class_weights(
    y: np.ndarray,
    method: str = 'inverse_frequency'
) -> Dict[int, float]:
    """
    Calculate class weights using inverse frequency.
    
    The weights are computed such that class_weight[i] * class_count[i] is
    approximately equal for all classes, ensuring balanced contribution to loss.
    
    Args:
        y: Label array. Can be:
            - 1D array of class indices (e.g., [0, 1, 2, 0, ...])
            - 2D one-hot encoded array (e.g., [[1,0,0], [0,1,0], ...])
        method: Weighting method. Currently supports 'inverse_frequency'.
        
    Returns:
        Dict[int, float]: Dictionary mapping class index to weight.
        
    Raises:
        ValueError: If unsupported method is specified or y is empty.
        
    Example:
        >>> y = np.array([0, 0, 0, 1, 2])  # 3 class 0, 1 class 1, 1 class 2
        >>> weights = compute_class_weights(y)
        >>> # weights will give higher weight to minority classes
    """
    if len(y) == 0:
        raise ValueError("Cannot compute class weights for empty array")
    
    if method != 'inverse_frequency':
        raise ValueError(
            f"Unsupported method: {method}. "
            "Supported methods: 'inverse_frequency'"
        )
    
    # Convert one-hot to class indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    
    # Get unique classes and their counts
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    
    # Compute inverse frequency weights
    # Formula: weight[i] = n_samples / (n_classes * count[i])
    # This ensures: weight[i] * count[i] = n_samples / n_classes (constant)
    weights = {}
    for cls, count in zip(classes, counts):
        weights[int(cls)] = n_samples / (n_classes * count)
    
    return weights


def get_class_distribution(y: np.ndarray) -> Dict[str, Union[Dict, int]]:
    """
    Analyze and return class distribution statistics.
    
    Args:
        y: Label array (1D class indices or 2D one-hot encoded).
        
    Returns:
        Dict containing:
            - 'counts': Dict mapping class index to count
            - 'percentages': Dict mapping class index to percentage
            - 'total': Total number of samples
    """
    # Convert one-hot to class indices if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    return {
        'counts': {int(cls): int(cnt) for cls, cnt in zip(classes, counts)},
        'percentages': {int(cls): float(cnt / total * 100) for cls, cnt in zip(classes, counts)},
        'total': total
    }


def compute_alpha_from_weights(
    class_weights: Dict[int, float],
    n_classes: int = 3
) -> List[float]:
    """
    Convert class weights to alpha values for focal loss.
    
    Normalizes class weights so they sum to 1, suitable for use as
    alpha parameter in focal loss.
    
    Args:
        class_weights: Dictionary mapping class index to weight.
        n_classes: Number of classes.
        
    Returns:
        List[float]: Normalized alpha values for each class.
    """
    # Get weights in order
    weights = [class_weights.get(i, 1.0) for i in range(n_classes)]
    
    # Normalize to sum to 1
    total = sum(weights)
    alpha = [w / total for w in weights]
    
    return alpha


# TensorFlow-dependent functions (for Google Colab)
# These will only be available when TensorFlow is installed

def get_focal_loss(
    gamma: float = 2.0,
    alpha: Optional[List[float]] = None,
    from_logits: bool = False
):
    """
    Create a focal loss function for imbalanced classification.
    
    Focal loss down-weights well-classified examples and focuses on hard,
    misclassified examples. This is particularly useful for extreme class
    imbalance where the model might otherwise focus on the majority class.
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
        - p_t is the probability of the correct class
        - gamma is the focusing parameter (higher = more focus on hard examples)
        - alpha_t is the class weight for the true class
    
    Args:
        gamma: Focusing parameter. Higher values increase focus on hard examples.
            - gamma=0: equivalent to cross-entropy
            - gamma=2: commonly used default
        alpha: Optional list of class weights [alpha_0, alpha_1, alpha_2].
            If None, all classes are weighted equally.
        from_logits: Whether predictions are logits (pre-softmax) or probabilities.
        
    Returns:
        tf.keras.losses.Loss: Focal loss function compatible with Keras.
        
    Raises:
        ImportError: If TensorFlow is not installed.
        
    Example:
        >>> focal_loss = get_focal_loss(gamma=2.0, alpha=[0.25, 0.5, 0.25])
        >>> model.compile(optimizer='adam', loss=focal_loss)
        
    References:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for focal loss. "
            "In Google Colab, TensorFlow is pre-installed. "
            "For local development, install with: pip install tensorflow"
        )
    
    class FocalLoss(tf.keras.losses.Loss):
        """Custom focal loss implementation for multi-class classification."""
        
        def __init__(
            self,
            gamma: float = 2.0,
            alpha: Optional[List[float]] = None,
            from_logits: bool = False,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.gamma = gamma
            self.alpha = alpha
            self.from_logits = from_logits
        
        def call(self, y_true, y_pred):
            """
            Compute focal loss.
            
            Args:
                y_true: Ground truth labels (one-hot encoded).
                y_pred: Predicted probabilities or logits.
                
            Returns:
                tf.Tensor: Scalar loss value.
            """
            # Convert logits to probabilities if needed
            if self.from_logits:
                y_pred = tf.nn.softmax(y_pred, axis=-1)
            
            # Clip predictions to prevent log(0)
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Get the probability of the true class
            # y_true is one-hot, so this extracts p_t for each sample
            p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
            
            # Compute focal weight: (1 - p_t)^gamma
            focal_weight = tf.pow(1.0 - p_t, self.gamma)
            
            # Compute cross-entropy: -log(p_t)
            ce = -tf.math.log(p_t)
            
            # Apply focal weight
            focal_loss = focal_weight * ce
            
            # Apply class weights (alpha) if provided
            if self.alpha is not None:
                alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
                # Get alpha for true class
                alpha_t = tf.reduce_sum(y_true * alpha_tensor, axis=-1)
                focal_loss = alpha_t * focal_loss
            
            return tf.reduce_mean(focal_loss)
        
        def get_config(self) -> Dict:
            """Return config for serialization."""
            config = super().get_config()
            config.update({
                'gamma': self.gamma,
                'alpha': self.alpha,
                'from_logits': self.from_logits
            })
            return config
    
    return FocalLoss(gamma=gamma, alpha=alpha, from_logits=from_logits)

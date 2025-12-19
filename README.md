# Industrial Pump Predictive Maintenance

**Course:** 62FIT4ATI - Artificial Intelligence
**Group:** 29
**Topic:** 2 - Recurrent Neural Network for Predictive Maintenance

---

## Project Overview

A deep learning solution using GRU (Gated Recurrent Unit) networks to predict industrial pump failures from sensor data. The system detects anomalies before failures occur, enabling proactive maintenance.

### Key Results

| Model | Balanced Accuracy | ROC-AUC |
|-------|-------------------|---------|
| 3-Class (NORMAL/RECOVERING/BROKEN) | 62.46% | - |
| Binary (NORMAL/ANOMALY) | **93.11%** | **96.11%** |

---

## Repository Structure

```
.
├── 62FIT4ATI_Group_29_Topic_2.ipynb              # Main notebook (run this)
├── README.md                                      # This file
├── sensor.csv                                     # Dataset (220,320 samples)
├── models/                                        # Trained model files
│   ├── pump_3class_model.keras                   #   3-class classifier
│   ├── pump_binary_model.keras                   #   Binary classifier
│   ├── scaler.pkl                                #   Data scaler
│   ├── label_encoder.pkl                         #   Label encoder
│   └── config.pkl                                #   Model configuration
└── reports/                                       # Written report
    ├── ATI_Report.md                              #   Markdown version
    └── ATI_Report.pdf                             #   PDF version
```

## Methodology

### Techniques Used

| Technique | Purpose |
|-----------|---------|
| GRU (32 units) | Capture temporal patterns in sensor data |
| Undersampling | Balance training data |
| Class Weights | Increase importance of minority classes |
| Gradient Clipping | Prevent exploding gradients |
| Early Stopping | Prevent overfitting |

---

## Model Files

Pre-trained models are saved in `models/` directory:

| File | Description |
|------|-------------|
| `pump_3class_model.keras` | 3-class classifier (NORMAL/RECOVERING/BROKEN) |
| `pump_binary_model.keras` | Binary classifier (NORMAL/ANOMALY) |
| `scaler.pkl` | StandardScaler for feature normalization |
| `label_encoder.pkl` | Label encoder for target classes |
| `config.pkl` | Model configuration (sequence length, features) |

### Loading Pre-trained Models

```python
import pickle
from tensorflow.keras.models import load_model

# Load models
model_binary = load_model('models/pump_binary_model.keras')
model_3class = load_model('models/pump_3class_model.keras')

# Load preprocessing objects
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/config.pkl', 'rb') as f:
    config = pickle.load(f)
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## Authors

**Group 29** - 62FIT4ATI Artificial Intelligence


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
├── 62FIT4ATI_Group_29_Topic_2_TimeSeries.ipynb   # Main notebook (run this)
├── README.md                                      # This file
├── sensor.csv                                     # Dataset (220,320 samples)
├── models/                                        # Trained model files
│   ├── pump_3class_model.keras                   #   3-class classifier
│   ├── pump_binary_model.keras                   #   Binary classifier
│   ├── scaler.pkl                                #   Data scaler
│   ├── label_encoder.pkl                         #   Label encoder
│   └── config.pkl                                #   Model configuration
├── reports/                                       # Written report
│   └── 62FIT4ATI_Group_29_Report.md              #   Analysis report (3-5 pages)
└── notebooks/                                     # Additional notebooks (if any)
```

---

## Quick Start

### Option 1: Google Colab (Recommended)

1. Upload `sensor.csv` to Google Drive (root of MyDrive)
2. Open `62FIT4ATI_Group_29_Topic_2_TimeSeries.ipynb` in Colab
3. Run all cells from top to bottom

### Option 2: Local Environment

```bash
# 1. Clone repository
git clone <repository-url>
cd Final

# 2. Install dependencies
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn

# 3. Run Jupyter notebook
jupyter notebook 62FIT4ATI_Group_29_Topic_2_TimeSeries.ipynb
```

---

## Dataset

- **Source:** Industrial pump sensor readings
- **Samples:** 220,320
- **Features:** 52 sensors
- **Target Classes:**
  - NORMAL: 219,420 (99.59%)
  - RECOVERING: 893 (0.41%)
  - BROKEN: 7 (0.003%)

**Challenge:** Extreme class imbalance - only 7 failure samples!

---

## Methodology

### Techniques Used

| Technique | Purpose |
|-----------|---------|
| GRU (32 units) | Capture temporal patterns in sensor data |
| Undersampling | Balance training data |
| Class Weights | Increase importance of minority classes |
| Gradient Clipping | Prevent exploding gradients |
| Early Stopping | Prevent overfitting |

### Why NOT SMOTE?

SMOTE creates synthetic samples by interpolation. For time-series data, this breaks temporal dependencies and creates unrealistic patterns.

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

---

## Results Summary

### Binary Model (Recommended for Production)

- **Balanced Accuracy:** 93.11%
- **ROC-AUC:** 96.11%
- **Anomaly Recall:** 88.2%

The Binary model is recommended for production use because:
1. Higher balanced accuracy
2. Better discrimination (ROC-AUC)
3. Simpler decision: "Is something wrong?" vs "What exactly is wrong?"

### 3-Class Model

- **Balanced Accuracy:** 62.46%
- Cannot distinguish BROKEN from RECOVERING (insufficient BROKEN samples)

---

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

---

## License

This project is for educational purposes as part of the 62FIT4ATI course.

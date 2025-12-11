# Industrial Pump Predictive Maintenance

An RNN-based predictive maintenance system for industrial pumps using LSTM architecture to analyze time-series sensor data and classify machine status.

## Project Overview

This project builds a deep learning model to predict industrial pump failures based on 52 sensor measurements. The system classifies machine status into three categories:
- **NORMAL**: Pump operating normally
- **RECOVERING**: Pump recovering from an issue
- **BROKEN**: Pump has failed

### Key Challenges
- Extreme class imbalance (NORMAL: 205,836 | RECOVERING: 14,477 | BROKEN: 7)
- Temporal pattern recognition in sensor data
- Training stability for RNN models

## Project Structure

```
industrial-pump-maintenance/
├── notebooks/                  # Jupyter/Colab notebooks
│   └── 62FIT4ATI_Group_X_Topic_2.ipynb
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessor.py        # Data preprocessing
│   ├── imbalance_handler.py   # Class imbalance handling
│   ├── model_builder.py       # LSTM model architecture
│   ├── training_manager.py    # Training pipeline
│   ├── evaluator.py           # Model evaluation
│   └── inference.py           # Inference pipeline
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_model.py
│   └── test_inference.py
├── models/                     # Saved model files
├── reports/                    # LaTeX reports and figures
├── sensor.csv                  # Dataset
├── requirements.txt            # Python dependencies
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster training

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd industrial-pump-maintenance
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place the `sensor.csv` dataset in the project root directory.

### Google Colab Setup

1. Upload the notebook `62FIT4ATI_Group_X_Topic_2.ipynb` to Google Colab
2. Upload `sensor.csv` to your Google Drive
3. Run the setup cell to mount Google Drive and install dependencies
4. Enable GPU runtime: Runtime → Change runtime type → GPU (T4)

## Running the Notebook

### In Google Colab
1. Open the notebook in Colab
2. Run cells sequentially from top to bottom
3. The notebook follows this structure:
   - Section 1: Problem Formulation
   - Section 2: Identify Inputs and Outputs
   - Section 3: Data Preparation
   - Section 4: Optimization Techniques
   - Section 5: Neural Network Model
   - Section 6: Performance Measurement
   - Section 7: Inference on New Data
   - Section 8: Conclusion

### Locally with Jupyter
```bash
jupyter notebook notebooks/62FIT4ATI_Group_X_Topic_2.ipynb
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessor.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Model Architecture

The system uses a stacked LSTM architecture:
- Input: Sequences of 60 time steps × 52 sensor features
- LSTM Layer 1: 128 units with return sequences
- Dropout: 0.3
- LSTM Layer 2: 64 units
- Dropout: 0.3
- Dense: 32 units with ReLU
- Output: 3 units with Softmax (NORMAL, RECOVERING, BROKEN)

## Class Imbalance Handling

Due to extreme class imbalance, the following techniques are applied:
- Inverse frequency class weights
- Focal loss function (γ=2.0)
- Stratified train/validation/test splits

## Dependencies

- TensorFlow >= 2.12.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- imbalanced-learn >= 0.10.0
- hypothesis >= 6.70.0 (for property-based testing)

## License

This project is for educational purposes as part of the 62FIT4ATI course.

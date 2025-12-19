# Project 2 Report: GRU-Based Predictive Maintenance for Industrial Pump Systems

**Course:** 62FIT4ATI - Advanced Topic in Information Technology
**Group:** 29
**Class:** 03
**Members:** Nguyen Trung Hieu, Tran Nguyen Khai, Dao Viet Anh
**Lecturer:** Nguyen Xuan Thang
**Semester:** Fall 2025

---

## 1. Introduction

This project addresses predictive maintenance for industrial pump systems using sensor data. The objective is to classify machine state as NORMAL, BROKEN, or RECOVERING based on readings from 52 sensors (220,320 samples).

The primary challenge is **extreme class imbalance**: only 7 BROKEN samples (0.003%). We employ a dual-model strategy using GRU: a 3-class classifier and a binary classifier (NORMAL vs ANOMALY). This report analyzes and compares both approaches.

---

## 2. Model and Setup

### 2.1 Problem Formulation and Data Characteristics

The task is multivariate time-series classification using sliding windows of 20 timesteps, with labels corresponding to the machine state at the final timestep.

| Status | Count | Percentage |
|--------|-------|------------|
| NORMAL | 219,420 | 99.59% |
| RECOVERING | 893 | 0.41% |
| BROKEN | 7 | 0.003% |

The dataset exhibits extreme class imbalance characteristic of real-world predictive maintenance. During preprocessing, `sensor_04` was removed due to data leakage (-0.916 correlation with target). Analysis of BROKEN events revealed that sensors deviate 10-20 minutes before failure, confirming early warning potential.

### 2.2 Choice of Recurrent Architecture

A GRU-based architecture is selected for this task. Compared to LSTM, GRU offers:

- **Fewer parameters**: GRU has 2 gates versus LSTM's 3 gates, reducing model complexity
- **Faster training**: Simpler architecture leads to reduced computational cost
- **Less prone to overfitting**: Critical when training data for minority classes is extremely limited
- **Comparable performance**: For many sequence modeling tasks, GRU achieves similar results to LSTM

### 2.3 Dual-Model Strategy: Why Two Models?

**Model 1: 3-Class Classifier (NORMAL / RECOVERING / BROKEN)**
- Follows the original problem formulation directly
- Challenge: Only 7 BROKEN samples in entire dataset (4 in training)

**Model 2: Binary Classifier (NORMAL vs ANOMALY)**
- Combines BROKEN and RECOVERING into single ANOMALY class
- Increases minority class samples from 7 to 900
- Rationale: Detecting *any* anomaly is more actionable than distinguishing failure types

### 2.4 Network Architecture

| Layer | Configuration | Parameters |
|-------|---------------|------------|
| Input | (20 timesteps, 50 features) | -- |
| GRU | 32 units | 8,160 |
| Dropout | Rate = 0.3 | -- |
| Dense | 16 units, ReLU | 528 |
| Dropout | Rate = 0.2 | -- |
| Output | Softmax (3 or 2 classes) | 51 / 34 |
| **Total** | | ~8,700 |

Training: Adam optimizer (lr=0.0005), class-weighted loss, early stopping, undersampling. SMOTE avoided as it breaks temporal dependencies.

---

## 3. Results

### 3.1 Quantitative Performance Comparison

| Metric | 3-Class Model | Binary Model |
|--------|---------------|--------------|
| Accuracy | 98.01% | 98.03% |
| Balanced Accuracy | 62.46% | **93.11%** |
| F1 Score (Macro) | 40.08% | 21.44% |
| ROC-AUC | N/A | **96.11%** |

While both models achieve similar standard accuracy (~98%), this metric is misleading due to class imbalance—a trivial classifier predicting all samples as NORMAL would achieve 99.6% accuracy. The **balanced accuracy** reveals the true difference: the binary model (93.11%) dramatically outperforms the 3-class model (62.46%). This 30+ percentage point gap demonstrates that combining minority classes into a single ANOMALY category provides sufficient training data for effective learning.

### 3.2 Per-Class Performance

| Class | 3-Class True | 3-Class Pred | 3-Class Recall | Binary True | Binary Pred | Binary Recall |
|-------|--------------|--------------|----------------|-------------|-------------|---------------|
| NORMAL | 24,905 | 24,425 | 98.0% | 24,905 | 24,432 | 98.1% |
| RECOVERING | 75 | 556 | 89.3% | -- | -- | -- |
| BROKEN | 1 | 0 | **0.0%** | -- | -- | -- |
| ANOMALY | -- | -- | -- | 76 | 549 | **88.2%** |

The 3-class model achieves 0% recall on BROKEN because with only 4 training samples, the model cannot learn meaningful patterns. It classifies all anomalies as RECOVERING instead.

The binary model solves this by merging BROKEN and RECOVERING into ANOMALY, increasing minority class samples from 7 to 900. This enables 88.2% anomaly recall (67 of 76 anomalies detected).

### 3.3 ROC Curve Analysis

The ROC-AUC of **0.961** indicates excellent discrimination ability between normal and anomalous states. The model achieves 85% true positive rate at only 2% false positive rate.

### 3.4 Confusion Matrix Analysis

| | Pred: NORMAL | Pred: ANOMALY |
|---|--------------|---------------|
| **True: NORMAL** | 24,432 | 473 |
| **True: ANOMALY** | 9 | 67 |

The model predicts 549 anomalies when only 76 exist, yielding low precision (12.2%) but high recall (88.2%). Only 9 true anomalies were misclassified as NORMAL.

This trade-off is **acceptable for predictive maintenance**: the cost of 473 unnecessary inspections is far lower than missing 9 potential failures. A missed pump failure can cause production downtime, equipment damage, and safety hazards.

---

## 4. Discussion

### 4.1 Impact of the Dual-Model Approach

1. **Sample size matters**: The 3-class model fails on BROKEN (7 samples) but succeeds on RECOVERING (893 samples). The binary model succeeds by combining them (900 samples).

2. **Problem reformulation is valid**: When the original problem formulation is infeasible due to data constraints, reformulating to a more practical task yields better results.

3. **Practical vs theoretical**: The 3-class approach is theoretically correct but practically useless for BROKEN detection.

### 4.2 Limitations

1. **Extremely limited failure data**: Only 7 BROKEN events constrain model learning
2. **High false positive rate**: 473 false alarms may cause alert fatigue
3. **Single failure mode**: Results may not generalize to different pump failure types
4. **Class ambiguity**: RECOVERING overlaps with early BROKEN stages

---

## 5. Conclusion

| Metric | 3-Class | Binary |
|--------|---------|--------|
| Balanced Accuracy | 62.46% | **93.11%** |
| Anomaly Recall | 44.7% | **88.2%** |
| BROKEN Detection | 0% | N/A |
| ROC-AUC | N/A | **96.11%** |

**Key findings**:
1. The 3-class model fails completely on BROKEN class (0% recall) due to insufficient training samples
2. The binary model achieves 93.11% balanced accuracy and 96.11% ROC-AUC by reformulating the problem
3. High false positive rate is acceptable given cost asymmetry in maintenance contexts
4. Problem reformulation (3-class → binary) is a valid strategy when original formulation is data-constrained

Future work could focus on collecting more failure data and exploring unsupervised anomaly detection approaches.

---

**62FIT4ATI Class 03 - Group 29 - Topic 2**

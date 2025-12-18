# Industrial Pump Predictive Maintenance Using Recurrent Neural Networks
## Results Analysis Report

**Course:** 62FIT4ATI - Artificial Intelligence  
**Group:** 29  
**Topic:** 2 - Recurrent Neural Network for Predictive Maintenance

---

## 1. Introduction

This report presents the analysis of our RNN-based predictive maintenance system for industrial pumps. The system was trained on sensor data from 52 sensors (220,320 samples) to predict pump failures, classifying pump status into three states: NORMAL, RECOVERING, and BROKEN.

The primary challenge was extreme class imbalance - only **7 BROKEN samples** (0.003%) in the entire dataset. We developed two models: a 3-class classifier and a binary classifier (NORMAL vs ANOMALY).

---

## 2. Data Analysis Results

### 2.1 Class Distribution

| Status | Count | Percentage |
|--------|-------|------------|
| NORMAL | 219,420 | 99.59% |
| RECOVERING | 893 | 0.41% |
| BROKEN | 7 | 0.003% |

The dataset exhibits **extreme class imbalance** - BROKEN samples represent only 0.003% of total data. This is typical in predictive maintenance where failures are rare but critical events.

### 2.2 Sensor Correlation Analysis

Key findings from correlation analysis:

| Finding | Details |
|---------|---------|
| Data Leakage Detected | `sensor_04` has -0.916 correlation with target (removed from features) |
| High Inter-sensor Correlation | Multiple sensor pairs show >0.9 correlation |
| Useful Predictors | Sensors 00, 06, 15, 38, 50 show moderate correlation with machine status |

### 2.3 Temporal Patterns Around Failures

Analysis of the 7 BROKEN events revealed consistent patterns:

```
Sensor Behavior Timeline:
─────────────────────────────────────────────────────
  -30 min      -15 min       0 min        +15 min
     │            │            │             │
     └─ Normal    └─ Gradual   └─ BROKEN    └─ RECOVERING
        readings     drift        event        phase
                     begins
```

**Key observations:**
- Sensors begin deviating from normal **10-20 minutes before** BROKEN label
- `sensor_00` and `sensor_06` show earliest warning signs
- RECOVERING phase follows immediately after BROKEN, lasting 30-60 minutes
- Sensor values during RECOVERING overlap significantly with early BROKEN stages

### 2.4 Missing Data Analysis

- No missing values in sensor readings
- All 220,320 timestamps have complete sensor data
- Dataset is clean and ready for modeling without imputation

---

## 3. Model Results and Analysis

### 3.1 Model Performance Comparison

| Metric | 3-Class Model | Binary Model |
|--------|---------------|--------------|
| Accuracy | 98.01% | 98.03% |
| Balanced Accuracy | 62.46% | **93.11%** |
| F1 Score | 40.08% | 21.44% |
| ROC-AUC | N/A | **96.11%** |

### 3.2 Per-Class Analysis

#### 3-Class Model Results:
| Class | True Count | Predicted | Recall |
|-------|------------|-----------|--------|
| NORMAL | 24,905 | 24,425 | 98.0% |
| RECOVERING | 75 | 556 | 89.3% |
| BROKEN | 1 | 0 | 0.0% |

#### Binary Model Results:
| Class | True Count | Predicted | Recall |
|-------|------------|-----------|--------|
| NORMAL | 24,905 | 24,432 | 98.1% |
| ANOMALY | 76 | 549 | 88.2% |

### 3.3 Key Findings

**Finding 1: Binary model significantly outperforms 3-class model for anomaly detection**
- Balanced Accuracy: 93.11% vs 62.46%
- ROC-AUC of 96.11% indicates excellent discrimination ability between normal and anomalous states

**Finding 2: 3-Class model fails to distinguish BROKEN from RECOVERING**
- 0% recall on BROKEN class (1 sample in test set, 0 correctly predicted)
- Only 4 BROKEN samples in training data is insufficient for learning
- Model learns to classify all anomalies as RECOVERING

**Finding 3: High false positive rate is acceptable in this context**
- Binary model predicts 549 anomalies when only 76 exist
- In predictive maintenance, false alarms are preferable to missed failures
- Cost of unnecessary inspection << Cost of unexpected breakdown

### 3.4 ROC Curve Analysis

The Binary model achieves ROC-AUC of **0.961**, indicating:
- Strong separation between NORMAL and ANOMALY classes
- Threshold can be adjusted based on operational requirements:
  - Lower threshold → Higher recall (catch more failures) but more false alarms
  - Higher threshold → Fewer false alarms but risk missing failures

### 3.5 Confusion Matrix Interpretation

**3-Class Model:**
- High precision for NORMAL class (98%)
- RECOVERING detection works reasonably (89.3% recall)
- Complete failure on BROKEN class due to data scarcity

**Binary Model:**
- Effectively combines BROKEN + RECOVERING into ANOMALY
- 88.2% recall means only ~12% of anomalies are missed
- Trade-off: 473 false positives (NORMAL predicted as ANOMALY)

---

## 4. Discussion

### 4.1 Why Binary Classification Works Better

The binary approach succeeds because:
1. **More training samples**: Combining BROKEN (4) + RECOVERING (75) gives 79 anomaly samples vs just 4 BROKEN
2. **Clearer decision boundary**: NORMAL vs "not NORMAL" is easier to learn than distinguishing subtle differences between RECOVERING and BROKEN
3. **Practical relevance**: In real operations, any anomaly requires attention regardless of severity

### 4.2 Early Warning System Potential

Analysis of sensor behavior revealed that sensors change **10-20 minutes BEFORE** the BROKEN label appears:

```
Timeline of Failure Event:
───────────────────────────────────────────────
-20 min     -10 min      0 min       +10 min
   │           │           │            │
   └─ Sensors  └─ Sensors  └─ BROKEN   └─ RECOVERING
      start       clearly     label        begins
      changing    abnormal    appears
```

This means our model can potentially provide **early warning** before actual failure occurs, giving maintenance teams time to respond.

### 4.3 Limitations

1. **Extremely limited failure samples**: Only 7 BROKEN events limits model's ability to learn failure patterns
2. **Single failure mode**: Results may not generalize to different types of pump failures
3. **High false positive rate**: 473 false alarms may cause "alert fatigue" in production
4. **Class ambiguity**: RECOVERING state overlaps with early BROKEN stages, making distinction difficult

### 4.4 Recommendations for Deployment

| Scenario | Recommended Model | Threshold | Expected Behavior |
|----------|-------------------|-----------|-------------------|
| High-risk equipment | Binary | Low (0.3) | Catch all anomalies, accept more false alarms |
| Standard monitoring | Binary | Medium (0.5) | Balanced approach |
| Low-risk equipment | Binary | High (0.7) | Fewer alerts, may miss some anomalies |

---

## 5. Conclusion

Our RNN-based predictive maintenance system demonstrates that **Binary classification (NORMAL vs ANOMALY)** is more effective than multi-class classification for this problem. Despite extreme class imbalance (7 failure samples out of 220,320), we achieved:

- **93.11% Balanced Accuracy** with Binary classification
- **96.11% ROC-AUC** indicating excellent discrimination ability
- **88.2% Recall** for anomaly detection

The key insight is that in predictive maintenance, detecting that something is wrong is more valuable than precisely categorizing the type of failure. A simple GRU architecture with proper class imbalance handling proves effective even with severely limited failure data.

**Future work** should focus on collecting more failure data and exploring anomaly detection approaches (autoencoders) that learn normal patterns rather than relying on labeled failure samples.

---

**62FIT4ATI - Artificial Intelligence**  
**Group 29 - Topic 2: RNN for Predictive Maintenance**

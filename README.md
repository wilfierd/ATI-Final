# Industrial Pump Predictive Maintenance

Dự đoán sự cố máy bơm công nghiệp từ dữ liệu cảm biến.

## Bài toán

Phân loại trạng thái máy bơm (NORMAL / BROKEN / RECOVERING) dựa trên 52 sensors.

Khó khăn: Chỉ có **7 mẫu BROKEN** trong 220,320 mẫu. Cực kỳ mất cân bằng.

## Cách tiếp cận

Thay vì cố gắng phân biệt 3 class, mình dùng thêm Binary model (NORMAL vs ANOMALY) vì thực tế chỉ cần biết "máy có vấn đề hay không" là đủ để cảnh báo.

Không dùng SMOTE vì với time series, việc tạo synthetic samples sẽ tạo ra pattern giả.

## Kết quả

| Model | Balanced Accuracy | ROC-AUC |
|-------|-------------------|---------|
| 3-Class GRU | 62% | - |
| Binary GRU | **93%** | **96%** |

## Cách chạy

1. Upload `sensor.csv` lên Google Drive (thư mục gốc MyDrive)
2. Mở notebook `62FIT4ATI_Group_X_Topic_2_TimeSeries_(2).ipynb` trên Colab
3. Chạy hết từ trên xuống

Notebook tự nhận diện môi trường Colab hay Local, không cần config gì.

## Kỹ thuật chính

- **GRU** (32 units) - đơn giản, tránh overfit
- **Undersampling** - giảm NORMAL xuống bằng RECOVERING
- **Class weights** - tăng weight cho minority class (max 10x)
- **Gradient clipping** - ổn định training
- **Early stopping** - dừng khi val_loss không giảm

## Files

```
sensor.csv                    <- Dataset (cần tải về)
62FIT4ATI_..._TimeSeries_(2).ipynb  <- Notebook chính
models/                       <- Model đã train
```

## Ghi chú

Với 7 mẫu BROKEN, không model nào phân biệt được BROKEN vs RECOVERING. Nhưng Binary model detect ANOMALY rất tốt (93%), đủ để làm hệ thống cảnh báo sớm.

---
**62FIT4ATI - Artificial Intelligence**

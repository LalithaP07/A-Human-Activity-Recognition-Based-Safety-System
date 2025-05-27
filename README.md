# A-Human-Activity-Recognition-Based-Safety-System
# 🕵️‍♀️ Human Activity Recognition with Abnormal Activity Detection & SOS Alert System

This project combines **Human Activity Recognition (HAR)** using deep learning with a **safety system** capable of detecting abnormal activities (like falls or sudden movements) and sending **automated SOS alerts** via email or SMS.

---

## 📌 Project Overview

The goal is to build a safety system, especially for women and vulnerable individuals, that:

✅ Classifies normal daily activities (walking, standing, sitting, etc.)  
✅ Detects abnormal or dangerous movements (e.g., falls, sprints)  
✅ Sends an automated SOS alert with location data when danger is detected

---

## 📊 Dataset

**Source**: [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)

- Accelerometer and gyroscope data from smartphones
- 30 participants performing 6 activities
- Each activity recorded in 128 time-step windows
- Data segmented into:
  - `train/Inertial Signals/`
  - `test/Inertial Signals/`

---

## ⚙️ Technologies Used

- **Python**  
- **TensorFlow/Keras** (for LSTM and RNN models)  
- **NumPy, Pandas** (data handling)  
- **Scikit-learn** (metrics, preprocessing)  
- **Matplotlib, Seaborn** (visualization)  
- **Twilio / smtplib** (for SOS alerts)  
- **Geopy** (GPS location for alerts)

---

## 🧠 Model Architectures

### LSTM Model
```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(1, 561)),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(6, activation='softmax')
])
RNN
model = Sequential([
    SimpleRNN(128, return_sequences=True, input_shape=(1, 561)),
    Dropout(0.5),
    SimpleRNN(64),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

Results
✅ Activity Classification Accuracy: 92% (LSTM), 89% (RNN)
✅ Abnormal Activity Detection: >85% precision, minimal false positives
✅ SOS Alerts: Delivered via email/SMS in < 5 seconds

SOS Alert System
Threshold-based anomaly detection flags unusual activity
Sends SOS message with GPS coordinates
Integrated with:
Twilio API for SMS
SMTP for email alerts

Visualizations
Accuracy/Loss plots for LSTM and RNN models
Correlation heatmaps of HAR features
Real-time alert simulations



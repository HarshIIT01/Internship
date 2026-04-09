# Spacecraft Anomaly Detection System

##  Project Overview
A comprehensive deep learning-based anomaly detection system designed to analyze spacecraft telemetry data in real-time. This project implements multiple state-of-the-art neural network architectures to detect anomalies in complex multivariate spacecraft sensor data.

##  Project Details
- **Created:** June 27, 2025
- **Language:** Python
- **Status:** Active Development
- **Internship Project:** AIML (Artificial Intelligence & Machine Learning)

##  Key Features

### Multiple Deep Learning Approaches
1. **Standard Autoencoder** - Unsupervised reconstruction-based anomaly detection
2. **LSTM Autoencoder** - Time series pattern anomaly detection
3. **CNN-LSTM Hybrid** - Supervised classification with convolutional neural networks
4. **Variational Autoencoder (VAE)** - Probabilistic anomaly detection with latent space modeling

### Data Processing
- **Synthetic Data Generation:** Realistic spacecraft telemetry simulation
- **Feature Engineering:** 15 sensor features including:
  - Temperature sensors (4)
  - Pressure sensors (3)
  - Voltage readings (4 channels)
  - Current readings (2 channels)
  - Vibration data (2 axes)
- **Data Scaling:** StandardScaler normalization for model stability

### Real-Time Capabilities
- Online anomaly detection with sliding window approach
- Confidence scoring for detected anomalies
- Real-time monitoring simulation

##  System Architecture

```
Input Data (15 features)
    ↓
Data Preprocessing & Scaling
    ↓
├─→ Autoencoder → Reconstruction Error
├─→ LSTM Autoencoder → Temporal Pattern Analysis
├─→ CNN-LSTM Hybrid → Supervised Classification
└─→ VAE → Probabilistic Detection
    ↓
Anomaly Detection Results
    ↓
Evaluation & Visualization
```

##  Model Performance Metrics
- **Accuracy:** Classification accuracy on test set
- **Precision:** True positive rate among predicted anomalies
- **Recall:** Anomaly detection rate
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under the receiver operating characteristic curve

##  Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

##  Usage

### Basic Usage
```python
from spacecraft_anomaly_detection import SpacecraftAnomalyDetector

# Initialize detector
detector = SpacecraftAnomalyDetector(sequence_length=50, threshold_percentile=95)

# Train models
detector.train_models(X_train, y_train, X_val, y_val, epochs=100)

# Detect anomalies
results = detector.detect_anomalies(X_test, y_test)

# Evaluate performance
evaluation_results = detector.evaluate_models(results, y_test)

# Visualize results
detector.plot_results(results, evaluation_results, X_test, y_test)
```

### Run Full Pipeline
```bash
python spacecraft_anomaly_detection.py
```

##  Project Structure
```
Internship/
├── README.md                           # Project documentation
└── spacecraft_anomaly_detection.py    # Main implementation
```

##  Key Components

### SpacecraftAnomalyDetector Class
Main class implementing all anomaly detection models:
- `generate_synthetic_data()` - Creates realistic telemetry data
- `preprocess_data()` - Data normalization and scaling
- `create_sequences()` - Time series sequence generation
- `build_autoencoder()` - Constructs standard autoencoder
- `build_lstm_autoencoder()` - Constructs LSTM-based autoencoder
- `build_cnn_lstm_hybrid()` - Constructs CNN-LSTM hybrid model
- `build_variational_autoencoder()` - Constructs VAE model
- `train_models()` - Trains all four models
- `detect_anomalies()` - Performs anomaly detection
- `evaluate_models()` - Calculates performance metrics
- `plot_results()` - Visualizes detection results
- `real_time_monitoring()` - Simulates real-time monitoring

##  Output and Visualization
The system generates comprehensive visualizations including:
- Reconstruction error plots with anomaly thresholds
- Model performance comparison (F1-scores)
- Feature importance analysis
- Confusion matrices for classification metrics

##  Anomaly Detection Logic
- **Threshold-based:** Anomalies detected when reconstruction error exceeds percentile-based threshold
- **Probabilistic:** VAE provides probability distribution over latent space
- **Temporal:** LSTM captures time-series dependencies
- **Supervised:** CNN-LSTM uses labeled data for classification

##  Data Specifications
- **Normal Data Percentage:** ~95%
- **Anomaly Percentage:** ~5%
- **Anomaly Types:**
  - Temperature spike anomalies
  - Pressure drop anomalies
  - Voltage anomalies
  - Multi-system anomalies

##  Learning Outcomes
This project demonstrates:
- Deep learning for time series analysis
- Unsupervised anomaly detection techniques
- Supervised classification approaches
- Real-time system implementation
- Comprehensive model evaluation
- Data visualization and interpretation

##  Notes
- The system is designed for spacecraft telemetry but can be adapted for other multivariate time series data
- Early stopping and model checkpointing prevent overfitting
- All models trained on CPU/GPU with TensorFlow/Keras backend

##  Contributing
This is an internship project. Contributions and improvements are welcome!

##  References
- Autoencoder theory and applications
- LSTM networks for time series
- Convolutional Neural Networks
- Variational Autoencoders (VAE)
- Anomaly Detection in IoT systems

##  Contact
Project maintained by: HarshIIT01

---

**Last Updated:** April 9, 2026
**Repository:** https://github.com/HarshIIT01/Internship

# Anomaly Detection using Isolation Forest

This project implements an anomaly detection system using the Isolation Forest algorithm. The code is designed to identify anomalies in a dataset by analyzing specific numerical features.

## Overview

Anomalies are data points that differ significantly from the majority of the data. The Isolation Forest algorithm is particularly effective for this purpose, as it isolates anomalies rather than profiling normal data points. This implementation focuses on preprocessing, applying the Isolation Forest algorithm, and interpreting the results.

## Project Workflow

1. **Data Loading**:
   - Load the dataset from a CSV file using pandas.

2. **Data Preprocessing**:
   - Convert categorical features into numerical values using `LabelEncoder`.
   - Remove NULL values.
   - Select relevant numerical features for anomaly detection.

3. **Model Training**:
   - Train an `IsolationForest` model with specified parameters, including:
     - `n_estimators`: Number of trees in the forest.
     - `contamination`: Proportion of anomalies in the dataset.

4. **Anomaly Detection**:
   - Predict anomalies using the trained model.
   - Separate anomalies (`-1`) from normal data points (`1`).

5. **Results**:
   - Print the number of anomalies detected.
   - Save anomalies to a CSV file (optional).

## Features Used

The following features from the dataset were used for anomaly detection:

- `age`
- `fnlwgt`
- `capital-gain`
- `hours-per-week`
- `salary`

## How to Use

1. **Requirements**:
   - Python (3.7 or higher)
   - Required libraries: `pandas`, `scikit-learn`

2. **Setup**:
   - Place the dataset (`data.csv`) in the specified directory.

3. **Run the Script**:
   - Execute the Python script:
     ```bash
     python Anomaly_detection_using_isolation_forest.py
     ```

4. **Output**:
   - The script will display:
     - Number of anomalies detected.
     - Sample anomalies.
   - Anomalies will be saved to a file `anomalies.csv` (optional).

## Dependencies

Install the required Python libraries using:

```bash
pip install pandas scikit-learn

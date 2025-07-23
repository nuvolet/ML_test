'''
Scikit-Learn - Time Series Anomaly Detection

Use Case: Detecting anomalies in sensor data or mission telemetry
'''

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Load time series data
data = pd.read_csv('sensor_data.csv')
features = ['temperature', 'pressure', 'humidity', 'vibration']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# Anomaly detection
isolation_forest = IsolationForest(
    contamination=0.1,  # Expected proportion of anomalies
    random_state=42
)

# Fit and predict
anomalies = isolation_forest.fit_predict(X_scaled)
data['anomaly'] = anomalies

# Identify anomalous points
anomalous_points = data[data['anomaly'] == -1]
print(f"Detected {len(anomalous_points)} anomalies")
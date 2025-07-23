'''
Supervised Learning: Precipitation Nowcasting

Problem:
Predicting short-term precipitation patterns (0-6 hours ahead) based on weather radar data.

Real-World Application:
The National Weather Service uses similar ConvLSTM models to improve flash flood warnings by predicting precipitation intensity and movement.
These models can capture the spatiotemporal dynamics of storm systems better than traditional forecasting methods.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose
from sklearn.model_selection import train_test_split

# Load radar reflectivity data (typically in NetCDF format)
import xarray as xr

# Open historical radar dataset
radar_data = xr.open_dataset('/path/to/radar_data.nc')
reflectivity = radar_data['reflectivity'].values  # Shape: [time, lat, lon]

# Prepare sequences of radar images for input and target
def create_sequences(data, seq_length=10, pred_horizon=6):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_horizon):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(reflectivity)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape for ConvLSTM [samples, timesteps, rows, cols, channels]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], y_train.shape[3], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], y_test.shape[3], 1)

# Build ConvLSTM model for precipitation nowcasting
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                     input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], 1),
                     padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False))
model.add(BatchNormalization())
model.add(Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train[:, -1], epochs=20, batch_size=16,
                   validation_data=(X_test, y_test[:, -1]))

# Generate predictions
predictions = model.predict(X_test)

# Visualize results
def plot_prediction(actual, predicted, timestep=0):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot actual radar reflectivity
    im1 = ax1.imshow(actual[timestep, 0, :, :, 0], cmap='jet')
    ax1.set_title('Actual Reflectivity')
    fig.colorbar(im1, ax=ax1)
    
    # Plot predicted reflectivity
    im2 = ax2.imshow(predicted[timestep, :, :, 0], cmap='jet')
    ax2.set_title('Predicted Reflectivity')
    fig.colorbar(im2, ax=ax2)
    
    # Plot difference
    im3 = ax3.imshow(actual[timestep, 0, :, :, 0] - predicted[timestep, :, :, 0], cmap='RdBu')
    ax3.set_title('Difference (Actual - Predicted)')
    fig.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.show()

plot_prediction(y_test, predictions, timestep=0)

# Calculate performance metrics
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test[:, -1].flatten(), predictions.flatten())
print(f"Mean Squared Error: {mse:.4f}")
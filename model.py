import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Updated create_sequences function to handle insufficient samples
def create_sequences(data, labels, timesteps):
    Xs, ys = [], []
    # Only create sequences if there are enough samples
    if len(data) <= timesteps:
        raise ValueError(f"Not enough data points to create sequences with {timesteps} timesteps.")
    for i in range(len(data) - timesteps):
        Xs.append(data[i:(i + timesteps)].values)
        ys.append(labels.iloc[i + timesteps])
    return np.array(Xs), np.array(ys)


# Load dataset
data = pd.read_csv("tesla_data.csv")

# Use all columns except target_close and time_steps as features
features = data.columns.difference(['Time_Step', 'Target_Close']).tolist()
target = 'Target_Close'

# Extract features and target variable
X = data[features]
y = data[target]

# Define the number of timesteps (e.g., last 20 days for each sample)
timesteps = 10


# Perform train-test split, shuffle = False to maintain the order of the time series
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Call the function to reshape X_train and X_test
X_train, y_train = create_sequences(X_train, y_train, timesteps)
X_test, y_test = create_sequences(X_test, y_test, timesteps)

num_features = X_train.shape[2]

# Define the model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(timesteps, num_features)))
model.add(MaxPooling1D(pool_size=2))
# Remove Flatten layer to retain 3D shape for LSTM
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))\

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Calculate directional accuracy
direction_accuracy = np.mean((y_pred > y_test[:-1]) == (y_test[1:] > y_test[:-1]))
print(f"Directional Accuracy: {direction_accuracy * 100:.2f}%")
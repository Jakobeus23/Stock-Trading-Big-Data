from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("tesla_data_other.csv", sep=",")

# Remove unnecessary columns. volume does not help and is drastically bigger than all the other ones. Target_close_7d was used to create the target variable in the old model but now causes data leakage
df.drop(columns=['Volume', 'Target_Close_7d'], inplace=True, errors='ignore')

# Define features and target
X = df.drop(columns=['Price_Increase_7d'])

y = df['Price_Increase_7d']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=69, shuffle=False
)

# Min-max scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM
X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build the LSTM model
LSTM_model = Sequential()
LSTM_model.add(LSTM(units=128, return_sequences=True, input_shape=(1, X_train_scaled.shape[2])))
LSTM_model.add(Dropout(0.2))
LSTM_model.add(LSTM(units=64, return_sequences=False))
LSTM_model.add(Dropout(0.1))
LSTM_model.add(Dense(units=1, activation='sigmoid'))
LSTM_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(name='auc')])

# Train the model
history = LSTM_model.fit(X_train_scaled, y_train, epochs=150, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the model on the test data
test_loss, test_auc = LSTM_model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test AUC: {test_auc}")

# Make predictions
y_pred = LSTM_model.predict(X_test_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)

# Identify incorrect predictions
incorrect_predictions = y_test != y_pred_binary.flatten()

# Print incorrect predictions
incorrect_indices = y_test.index[incorrect_predictions]
incorrect_df = pd.DataFrame({
    'Actual': y_test[incorrect_indices],
    'Predicted': y_pred_binary.flatten()[incorrect_predictions],
    'Probability': y_pred.flatten()[incorrect_predictions]
}, index=incorrect_indices)

print(f"\nNumber of incorrect predictions: {len(incorrect_df)}")
print(incorrect_df)

# Optionally save incorrect predictions to a CSV
incorrect_df.to_csv("incorrect_predictions.csv")

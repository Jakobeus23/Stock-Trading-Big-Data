from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def print_heatmap(df):
    # Adjust figure size and font
    plt.figure(figsize=(12, 10))  # Increase the figure size for better readability

    # Generate the heatmap
    sns.heatmap(
        df.corr(),  # Compute correlation matrix
        annot=True,  # Display correlation values
        fmt=".2f",  # Format for correlation values
        cmap="coolwarm",  # Colormap
        cbar=True,  # Show color bar
        annot_kws={"size": 8}  # Adjust font size for annotations
    )

    # Add title and adjust labels
    plt.title("Correlation Heatmap", fontsize=16)
    plt.xticks(fontsize=10, rotation=45)  # Rotate and adjust x-axis labels
    plt.yticks(fontsize=10, rotation=0)  # Adjust y-axis labels
    plt.show()


def data_leakage_check():
    # Check if the target variable 'Price_Increase_7d' exists in the features
    if y in X:
        print(f"Potential data leakage! The target variable '{y}' is included as a feature.")
    else:
        print("No direct target leakage detected.")

    # Correlation check between features and target
    correlation_with_target = df[X + [y]].corr()[y].drop(y)  # Drop self-correlation
    print("Correlation of features with the target variable:")
    print(correlation_with_target)

    # Threshold for high correlation (potential leakage)
    threshold = 0.9
    high_correlation_features = correlation_with_target[correlation_with_target.abs() > threshold]

    if not high_correlation_features.empty:
        print("\nWARNING: The following features are highly correlated with the target:")
        print(high_correlation_features)
    else:
        print("\nNo features are highly correlated with the target. Data leakage unlikely.")

    # Check for overlapping derived features (e.g., rolling features or future-looking features)
    print("\nChecking for potentially derived features...")
    derived_features = ['Target_Close_7d', 'Price_Movement']  # Add any features you think might cause leakage
    leaking_features = [feature for feature in X if feature in derived_features]

    if leaking_features:
        print(f"WARNING: The following features might cause data leakage: {leaking_features}")
    else:
        print("No obvious derived features causing leakage were found.")




# read in dataset
df = pd.read_csv("tesla_data_other.csv", sep = ",")

# removing these columns as they are not needed for the model or cause a data leak
df.drop(columns = ['Volume'])
df.drop(columns = ['Target_Close_7d'])


X = df.drop(columns = ['Price_Increase_7d'])

# response variable
y = df['Price_Increase_7d']


# split the dataset into training and testing with an 80/20 split with a random state
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size = 0.2, random_state = 69, shuffle=False)


# min-max scaler - Xscaled = (X - Xmin) / (Xmax - Xmin) - may not be optimal for our data idrk
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# using numPy to reshape data. LSTM requires 3D data - samples (individual data points(rows)), time steps (length of time step used to predict target var), features (predictor columns used for making predictions)
# default of 1 time step, might set this to 10 or something.
X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))



# build LSTM m
LSTM_model = Sequential()

# adding first layer. Dropout sets a # of neurons to zero during training to prevent overreliance on certain patterns. (overfitting)
# "X_train.shape[1], X_train.shape[2]" selects the time steps and the # features as the 'input shape' so the model knows what data it's getting
LSTM_model.add(LSTM(units=128, return_sequences=True, input_shape=(1, 20)))

# Add Dropout to prevent overfitting
LSTM_model.add(Dropout(0.2))

# second layer, adds 50 more neurons
LSTM_model.add(LSTM(units=64, return_sequences=False))
LSTM_model.add(Dropout(0.1))


# output layer
LSTM_model.add(Dense(units=1, activation='sigmoid'))

# compiles model w selected features. The optimizer is how the model updates weights using backpropagation calc and is (variant of SGD - stochastic gradient descent) - loss function tells optimizer how far guesses were off.
LSTM_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(name='auc')])

# train the model
history = LSTM_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the model on test data
test_loss, test_auc = LSTM_model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test AUC: {test_auc}")

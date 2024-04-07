import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from tensorflow.keras.callbacks import TensorBoard
import time

# Load dataset
model_name='perceptron'
dataset_name="addition_dataset_2"
d_set = pd.read_csv('./../../Data/'+dataset_name+'.csv')
dset_features = d_set.copy()
dset_labels = dset_features.pop('result')
dset_features = np.array(dset_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dset_features, dset_labels, test_size=0.2, random_state=42)

# Define the model
dset_model_Perceptron = tf.keras.Sequential([
    layers.Dense(1)
])

# Compile the model
dset_model_Perceptron.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Define TensorBoard callback
tensorboard_callback = TensorBoard(log_dir='./../../Observation/'+model_name+'/'+model_name+dataset_name+'logs')

# Start timer for training time
start_time = time.time()

# Train the model with TensorBoard callback
history = dset_model_Perceptron.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# End timer for training time
training_time = time.time() - start_time

# Make predictions
predictions = dset_model_Perceptron.predict(X_test)

# Calculate metrics
# Calculate accuracy with threshold
threshold = 0.5  # Define the threshold
correct_predictions = np.sum(np.abs(y_test - predictions.flatten()) <= threshold)
total_predictions = len(y_test)
accuracy = correct_predictions / total_predictions  # Final accuracy
mse = mean_squared_error(y_test, predictions)  # Mean Squared Error
mae = mean_absolute_error(y_test, predictions)  # Mean Absolute Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
f1 = f1_score(y_test, predictions.round(), average='micro')  # F1 Score

# Calculate inference time
start_time = time.time()
predictions = dset_model_Perceptron.predict(X_test)
inference_time = time.time() - start_time

# Print and store metrics
metrics = {
    'Model type': model_name,
    'dataset': dataset_name,
    'Accuracy': accuracy,
    'MSE': mse,
    'MAE': mae,
    'RMSE': rmse,
    'F1 Score': f1,
    'Training Time': training_time,
    'Inference Time': inference_time
}

print("Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

# Store metrics in a list
metrics_list = list(metrics.values())
pd.DataFrame([metrics_list], columns=metrics.keys()).to_csv('./../../Observation/'+model_name+'/'+model_name+dataset_name+'metrics.csv', index=False)

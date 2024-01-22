"""
This code works with total data

"""
import numpy as np
import xarray as xr
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LSTM
import os
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#---------------------------------------------------------------------
# Function to read data and process
#---------------------------------------------------------------------
def get_data(test_size=0.2, random_state=0):
    file_path = 'Data_noaa_copernicus/noaa_avhrr/noaa_icesmi_combinefile.nc'
    ds = xr.open_dataset(file_path)

    days = ds['time'].values
    lat = ds['lat'].values
    lon = ds['lon'].values

    # Preprocess data, handling NaN values by avoiding them
    sst_data = ds['sst'].values

    # Find indices of non-NaN values
    non_nan_indices = np.where(~np.isnan(sst_data))

    # Extract non-NaN values
    sst_data = sst_data[non_nan_indices[0], non_nan_indices[1], non_nan_indices[2]]

    # Reshape the data to match the LSTM input shape
    sst_data = sst_data.reshape(sst_data.shape[0], -1)

    # Normalize the data
    scaler = MinMaxScaler()
    sst_data_normalized = scaler.fit_transform(sst_data)

    # Create sequences for time series prediction
    sequence_length = 10
    X_data, Y_data = [], []
    for i in range(len(sst_data_normalized) - sequence_length):
        X_data.append(sst_data_normalized[i:i+sequence_length])
        Y_data.append(sst_data_normalized[i+sequence_length])

    X_data, Y_data = np.array(X_data), np.array(Y_data)

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=test_size, random_state=random_state)

    # Return Feature and Target variables for training and validation
    return X_train, X_val, Y_train, Y_val

#---------------------------------------------------------------------
# Function to create the default configuration for the model. This will be overridden as 
# required during experimentation REGRESSION & LINEAR
#---------------------------------------------------------------------
def base_model_config():
    model_config = {
        "HIDDEN_NODES": [8, 16, 32],
        "HIDDEN_ACTIVATION": 'relu',
        "OUTPUT_NODES": 1,
        "OUTPUT_ACTIVATION": "linear",
        "WEIGHTS_INITIALIZER": "glorot_normal",
        "BIAS_INITIALIZER": "zeros",
        "NORMALIZATION": "batch",
        "OPTIMIZER": "adam",
        "LEARNING_RATE": 0.001,
        "REGULARIZER": None,
        "DROPOUT_RATE": 0.1,
        "EPOCHS": 10,
        "BATCH_SIZE": 32,
        "VALIDATION_SPLIT": 0.2,
        "VERBOSE": 0,
        "LOSS_FUNCTION": "mean_squared_error",
        "METRICS": ["mean_squared_error", "mean_absolute_error"]
    }
    return model_config

#---------------------------------------------------------------------
# Function to create a model and fit the model
#---------------------------------------------------------------------
def create_and_run_model(model_config, X, Y, model_name):
    model = keras.Sequential(name=model_name)
    
    for layer in range(len(model_config["HIDDEN_NODES"])):
        if layer == 0:
            model.add(
                LSTM(
                    model_config["HIDDEN_NODES"][layer],
                    return_sequences=True,
                    input_shape=(X.shape[1], X.shape[2]),
                    name="LSTM-Layer-" + str(layer),
                    kernel_initializer=model_config["WEIGHTS_INITIALIZER"],
                    bias_initializer=model_config["BIAS_INITIALIZER"],
                    activation=model_config["HIDDEN_ACTIVATION"]
                )
            )
        else:
            if model_config["NORMALIZATION"] == "batch":
                model.add(BatchNormalization())
                
            if model_config["DROPOUT_RATE"] > 0.0:
                model.add(Dropout(model_config["DROPOUT_RATE"]))
                
            model.add(
                LSTM(
                    model_config["HIDDEN_NODES"][layer],
                    activation=model_config["HIDDEN_ACTIVATION"],
                    return_sequences=True
                )
            )
                       
    model.add(
        LSTM(
            model_config["OUTPUT_NODES"],
            name="Output-Layer",
            activation=model_config["OUTPUT_ACTIVATION"]
        )
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=model_config["LEARNING_RATE"])
    
    model.compile(
        loss=model_config["LOSS_FUNCTION"],
        optimizer=optimizer,
        metrics=model_config["METRICS"]
    )
    
    print("\n******************************************************")
    model.summary()
    
    history = model.fit(
        X,
        Y,
        batch_size=model_config["BATCH_SIZE"],
        epochs=model_config["EPOCHS"],
        verbose=model_config["VERBOSE"],
        validation_split=model_config["VALIDATION_SPLIT"]
    )
    
    return history

# Function to plot a graph based on the results derived
def plot_graph(history, title):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 8))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=3)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=3)
        
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Initialize the measures
accuracy_measures = {}

for batch_size in range(32, 128, 16):

    # Load default configuration
    model_config = base_model_config()

    X_train, X_val, Y_train, Y_val = get_data()


    model_config["EPOCHS"] = 10
    # Set batch size to experiment value
    model_config["BATCH_SIZE"] = batch_size
    model_name = "Batch-Size-" + str(batch_size)
    history = create_and_run_model(model_config, X_train, Y_train, model_name)

    accuracy_measures[model_name] = history.history["mean_squared_error"]



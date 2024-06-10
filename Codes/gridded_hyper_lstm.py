import numpy as np
import xarray as xr
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

data = xr.open_dataset('Data_noaa_copernicus/noaa_avhrr/noaasst1982_2023_data.nc')
sst = data['sst'].values

# Extract time period for training, validation, and testing
train_start = np.datetime64('1982-01-01')
train_end = np.datetime64('2021-12-31')
test_start = np.datetime64('2022-01-01')
test_end = np.datetime64('2023-12-31')

train_data = sst[(data['time'] >= train_start) & (data['time'] <= train_end)]
test_data = sst[(data['time'] >= test_start) & (data['time'] <= test_end)]

# Replace NaN values with 0
train_data = np.nan_to_num(train_data)
test_data = np.nan_to_num(test_data)

# Normalize the data (fit on train data and transform both train and test data)
scaler = MinMaxScaler()
train_data_reshaped = train_data.reshape(train_data.shape[0], -1)
test_data_reshaped = test_data.reshape(test_data.shape[0], -1)

train_data_normalized = scaler.fit_transform(train_data_reshaped).reshape(train_data.shape)
test_data_normalized = scaler.transform(test_data_reshaped).reshape(test_data.shape)

# Flatten spatial dimensions, each row has 28*36 i.e 1008 values
train_data_flattened = train_data_normalized.reshape(train_data_normalized.shape[0], -1)
test_data_flattened = test_data_normalized.reshape(test_data_normalized.shape[0], -1)

# Define function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

# Define hyperparameters grid
# learning_rates = [0.001, 0.0001, 0.01]
# batch_sizes = [32, 64, 128]
# lstm_units = [(64, 32, 16), (128, 64, 32), (256, 128, 64)]
# dropout_rates = [0.1, 0.2, 0.3]
# epochs_list = [50, 100, 200]
# sequence_lengths = [30, 60, 90]

learning_rates = [0.0001]
batch_sizes = [64]
lstm_units = [(256, 128, 64)]
dropout_rates = [0.1]
epochs_list = [400]
sequence_lengths = [60]


best_rmse = float('inf')
best_config = {}

# Iterate over all hyperparameter combinations
for lr in learning_rates:
    for bs in batch_sizes:
        for units in lstm_units:
            for dr in dropout_rates:
                for epochs in epochs_list:
                    for seq_len in sequence_lengths:
                        # Update sequence length
                        train_sequences = create_sequences(train_data_flattened, seq_len)
                        test_sequences = create_sequences(test_data_flattened, seq_len)
                        x_train, x_val = train_test_split(train_sequences, test_size=0.2, random_state=42)

                        # Define model
                        model = Sequential([
                            LSTM(units[0], input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True),
                            LSTM(units[1], activation='relu', return_sequences=True),
                            LSTM(units[2], activation='relu', return_sequences=True),
                            Dropout(dr),
                            Dense(x_train.shape[2], activation='linear')
                        ])

                        # Compile model
                        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
                        
                        # Define early stopping callback
                        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        
                        # Train model
                        history = model.fit(x_train, x_train, epochs=epochs, batch_size=bs, validation_data=(x_val, x_val), verbose=1, callbacks=[early_stopping])

                        # Predict on test data
                        predicted_data = model.predict(test_sequences)
                        test_data_flat = test_sequences.reshape(-1, test_sequences.shape[-1])
                        predicted_data_flat = predicted_data.reshape(-1, predicted_data.shape[-1])
                        test_data_orig_flat = scaler.inverse_transform(test_data_flat)
                        predicted_data_orig_flat = scaler.inverse_transform(predicted_data_flat)
                        test_data_orig = test_data_orig_flat.reshape(test_sequences.shape)
                        predicted_data_orig = predicted_data_orig_flat.reshape(predicted_data.shape)
                        
                        # Calculate RMSE
                        mse = np.mean((test_data_orig - predicted_data_orig) ** 2)
                        rmse = np.sqrt(mse)
                        print(f"Config: lr={lr}, bs={bs}, units={units}, dr={dr}, epochs={epochs}, seq_len={seq_len}, RMSE={rmse}")

                        # Update best configuration
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_config = {
                                'learning_rate': lr,
                                'batch_size': bs,
                                'lstm_units': units,
                                'dropout_rate': dr,
                                'epochs': epochs,
                                'sequence_length': seq_len,
                                'rmse': rmse
                            }
                            model.save('Data_noaa_copernicus/noaa_avhrr/model_save/best_model.h5')

print(f"Best Configuration: {best_config}")

# Plot the best model's training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load the best model
#model = tf.keras.models.load_model('Data_noaa_copernicus/noaa_avhrr/model_save/best_model.h5')

predicted_data = model.predict(test_sequences)

# Inverse transform the predicted and test data back to original scale
test_data_flat = test_sequences.reshape(-1, test_sequences.shape[-1])
predicted_data_flat = predicted_data.reshape(-1, predicted_data.shape[-1])
test_data_orig_flat = scaler.inverse_transform(test_data_flat)
predicted_data_orig_flat = scaler.inverse_transform(predicted_data_flat)

# Reshape back to the original shape
test_data_orig = test_data_orig_flat.reshape(test_sequences.shape)
predicted_data_orig = predicted_data_orig_flat.reshape(predicted_data.shape)

# Calculate RMSE for each spatial grid point
rmse_grid = np.sqrt(np.mean((test_data_orig - predicted_data_orig) ** 2, axis=(0, 1)))
latitude = data['lat'].values
longitude = data['lon'].values


xticks = np.array([348, 350, 352, 354, 355])
yticks = np.array([50, 52, 54, 55.9])


plt.figure(figsize=(12, 6))

gs = gridspec.GridSpec(1, 3, width_ratios=[0.8, 0.8, 0.06])

ax1 = plt.subplot(gs[0])
img1 = ax1.imshow(test_data_orig[10, -1].reshape(sst.shape[1:]), cmap='jet', origin='lower', extent=[longitude.min(), longitude.max(), latitude.min(), latitude.max()])
ax1.set_title('Real SST')
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))

ax2 = plt.subplot(gs[1])
img2 = ax2.imshow(predicted_data_orig[10, -1].reshape(sst.shape[1:]), cmap='jet', origin='lower', extent=[longitude.min(), longitude.max(), latitude.min(), latitude.max()])
ax2.set_title('Predicted SST')
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))

cbar_ax = plt.subplot(gs[2])
plt.colorbar(img2, cax=cbar_ax)

plt.tight_layout()
plt.show()

# Plot RMSE spatial plot
plt.figure(figsize=(6, 6))

ax = plt.gca()
img = ax.imshow(rmse_grid.reshape(sst.shape[1:]), cmap='jet', origin='lower', extent=[longitude.min(), longitude.max(), latitude.min(), latitude.max()])
plt.colorbar(img)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
plt.title('RMSE')
plt.show()

